"""Train a graph neural network (chemprop D-MPNN) whose fingerprint is *learned*.

This is the "let the data define the fingerprint" half of the notebook. Unlike
MACCS/Morgan (features we impose) or even an MLP on Morgan bits (which would
smuggle the imposed lens back in), a D-MPNN learns its molecular representation
directly from the graph, driven only by the activity labels.

The model is multi-task: one shared learned fingerprint feeding two regression
heads, one per endpoint of a target pair (e.g. mu- and kappa-opioid pKi). A
single knob ``alpha`` weights the training loss toward endpoint A (alpha=1) or
endpoint B (alpha=0). Sliding it shows the payoff of the whole notebook: the
representation the data gives you depends on which question you ask it, and no
single learned fingerprint is best for both.

Training is fast (~seconds for a few epochs on ~2-4k molecules), but still too
slow / nondeterministic to run live per slider tick, so the notebook uses a
pre-trained grid of alpha values cached by ``scripts/train_alpha_grid.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

CACHE_MOLACE = Path(".cache/molace")

# Endpoint pairs available for the learned-fingerprint demo, keyed like the
# context-cliff target pairs so the notebook can reuse the same picker language.
ENDPOINT_PAIRS: dict[str, dict] = {
    "mu_vs_kappa": {
        "target_a": "mu-opioid",
        "dataset_a": "CHEMBL233_Ki",
        "target_b": "kappa-opioid",
        "dataset_b": "CHEMBL237_Ki",
    },
    "D3_vs_D4": {
        "target_a": "Dopamine D3",
        "dataset_a": "CHEMBL234_Ki",
        "target_b": "Dopamine D4",
        "dataset_b": "CHEMBL219_Ki",
    },
}


@dataclass(frozen=True)
class EndpointData:
    """Molecules + two-task pKi labels (NaN where a molecule lacks a label)."""

    pair_key: str
    target_a: str
    target_b: str
    smiles: list[str]
    y: np.ndarray  # (n, 2), columns [target_a, target_b], NaN for missing


def _load_target(dataset: str) -> dict[str, float]:
    df = pl.read_csv(CACHE_MOLACE / f"{dataset}.csv")
    out: dict[str, float] = {}
    for smi, y in zip(df["smiles"].to_list(), df["y [pEC50/pKi]"].to_list()):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and y is not None:
            out[Chem.MolToSmiles(mol)] = float(y)
    return out


def load_endpoint_data(pair_key: str) -> EndpointData:
    spec = ENDPOINT_PAIRS[pair_key]
    da = _load_target(spec["dataset_a"])
    db = _load_target(spec["dataset_b"])
    smiles = sorted(set(da) | set(db))
    y = np.array(
        [[da.get(s, np.nan), db.get(s, np.nan)] for s in smiles], dtype=float
    )
    return EndpointData(
        pair_key=pair_key,
        target_a=spec["target_a"],
        target_b=spec["target_b"],
        smiles=smiles,
        y=y,
    )


def scaffold_split(
    smiles: list[str], frac_train: float = 0.8, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Bemis-Murcko scaffold split so near-duplicate structures don't leak.

    Whole scaffold groups go entirely to train or test. Larger groups are
    assigned first (standard scaffold-split heuristic), the rest shuffled.
    """
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaf = ""
        groups.setdefault(scaf, []).append(i)

    rng = np.random.default_rng(seed)
    ordered = sorted(groups.values(), key=len, reverse=True)
    n_train_target = int(frac_train * len(smiles))
    train_idx: list[int] = []
    test_idx: list[int] = []
    for grp in ordered:
        if len(train_idx) + len(grp) <= n_train_target:
            train_idx.extend(grp)
        else:
            test_idx.extend(grp)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


@dataclass(frozen=True)
class TrainResult:
    alpha: float
    r2_a: float  # test R^2 on endpoint A
    r2_b: float  # test R^2 on endpoint B
    embedding: np.ndarray  # (n, d) learned fingerprint for all molecules
    smiles: list[str]


def train_dmpnn(
    ed: EndpointData,
    alpha: float,
    *,
    depth: int = 3,
    epochs: int = 30,
    seed: int = 0,
    accelerator: str = "cpu",
) -> TrainResult:
    """Train the 2-task D-MPNN with loss weight `alpha` on endpoint A.

    Returns test R^2 per endpoint and the learned fingerprint for every
    molecule. `alpha` in [0, 1]; task_weights = [alpha, 1 - alpha].
    """
    import lightning as L
    import torch
    from chemprop import data, featurizers, models, nn
    from chemprop.data import BatchMolGraph

    torch.manual_seed(seed)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_idx, test_idx = scaffold_split(ed.smiles, seed=seed)

    def _dps(indices):
        return [
            data.MoleculeDatapoint.from_smi(ed.smiles[i], ed.y[i]) for i in indices
        ]

    train_dset = data.MoleculeDataset(_dps(train_idx), featurizer)
    # Normalize targets on train, apply to the model's output transform.
    scaler = train_dset.normalize_targets()
    train_loader = data.build_dataloader(train_dset, batch_size=64, seed=seed)

    mp = nn.BondMessagePassing(depth=depth)
    ffn = nn.RegressionFFN(
        n_tasks=2,
        input_dim=mp.output_dim,
        task_weights=[alpha, 1.0 - alpha],
        output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
    )
    model = models.MPNN(mp, nn.MeanAggregation(), ffn)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader)

    # Predict on test, compute per-endpoint R^2 (ignoring NaN labels).
    model.eval()
    all_bmg = BatchMolGraph([featurizer(Chem.MolFromSmiles(s)) for s in ed.smiles])
    with torch.no_grad():
        preds = model(all_bmg).numpy(force=True)  # (n, 2)
        emb = model.fingerprint(all_bmg).numpy(force=True).astype(np.float32)

    def _r2(col: int) -> float:
        yt = ed.y[test_idx, col]
        yp = preds[test_idx, col]
        mask = ~np.isnan(yt)
        if mask.sum() < 5:
            return float("nan")
        yt, yp = yt[mask], yp[mask]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return TrainResult(
        alpha=alpha,
        r2_a=_r2(0),
        r2_b=_r2(1),
        embedding=emb,
        smiles=list(ed.smiles),
    )
