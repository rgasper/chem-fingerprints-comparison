"""CV + Tukey benchmark: does a *learned* D-MPNN fingerprint beat classical
*fixed* fingerprints at predicting activity on these endpoints?

For each single endpoint (mu, kappa, D3, D4):
  - Methods:
      * D-MPNN         - learns its own fingerprint from the graph (Apple MPS)
      * Morgan+RF      - ECFP r=2, 2048 bits    -> RandomForest
      * MACCS+RF       - 167-bit structural keys -> RandomForest
      * RDKitTopo+RF   - hashed path fingerprint -> RandomForest
      * AtomPair+RF    - atom-pair fingerprint   -> RandomForest
  - 5 repeats x 5 scaffold-grouped folds  = 25 held-out RMSE values per method
    (scaffold grouping => no near-duplicate leakage across folds).
  - Also report RMSE on the activity-CLIFF molecules only (the hard subset).
  - Per endpoint: one-way ANOVA + Tukey HSD across methods on the per-fold RMSE.

Heavy but self-contained; runs on this MacBook via MPS. Writes a JSON + a
human-readable summary the notebook's rigor note can cite.

Run:
  uv run python scripts/cv_tukey_benchmark.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator as fpg
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import f_oneway, tukey_hsd
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog("rdApp.*")

CACHE = Path(".cache/molace")
OUT_DIR = Path("data/benchmark")
N_REPEATS = 5
N_FOLDS = 5
DMPNN_EPOCHS = 30
RF_TREES = 300

# Single endpoints: (label, molace dataset).
ENDPOINTS = [
    ("mu-opioid", "CHEMBL233_Ki"),
    ("kappa-opioid", "CHEMBL237_Ki"),
    ("Dopamine D3", "CHEMBL234_Ki"),
    ("Dopamine D4", "CHEMBL219_Ki"),
]

_MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=2048)
_RDKIT = fpg.GetRDKitFPGenerator(fpSize=2048)
_ATOMPAIR = fpg.GetAtomPairGenerator(fpSize=2048)


def load_endpoint(dataset: str):
    """Return (smiles, y, cliff) for one endpoint, canonicalized + deduped."""
    df = pl.read_csv(CACHE / f"{dataset}.csv")
    smis, ys, cliffs = [], [], []
    seen = set()
    for row in df.iter_rows(named=True):
        mol = Chem.MolFromSmiles(row["smiles"])
        y = row["y [pEC50/pKi]"]
        if mol is None or y is None:
            continue
        cs = Chem.MolToSmiles(mol)
        if cs in seen:
            continue
        seen.add(cs)
        smis.append(cs)
        ys.append(float(y))
        cliffs.append(bool(row["cliff_mol"]) if "cliff_mol" in row else False)
    return smis, np.array(ys), np.array(cliffs, dtype=bool)


def scaffold_of(smi: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smi, includeChirality=False)
    except Exception:
        return smi


def scaffold_folds(smiles, n_folds, seed):
    """Group by Bemis-Murcko scaffold, then greedily bin whole scaffold groups
    into folds (largest-first) so no scaffold spans two folds."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[scaffold_of(s)].append(i)
    rng = np.random.default_rng(seed)
    order = list(groups.values())
    rng.shuffle(order)
    order.sort(key=len, reverse=True)
    fold_sizes = [0] * n_folds
    fold_idx = [[] for _ in range(n_folds)]
    for grp in order:
        j = int(np.argmin(fold_sizes))
        fold_idx[j].extend(grp)
        fold_sizes[j] += len(grp)
    return fold_idx


def fp_matrix(smiles, kind):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if kind == "morgan":
            fp = _MORGAN.GetFingerprintAsNumPy(m)
        elif kind == "rdkit_topo":
            fp = _RDKIT.GetFingerprintAsNumPy(m)
        elif kind == "atom_pair":
            fp = _ATOMPAIR.GetFingerprintAsNumPy(m)
        elif kind == "maccs":
            fp = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.int8)
        else:
            raise ValueError(kind)
        rows.append(fp)
    return np.asarray(rows, dtype=np.float32)


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if len(yt) else float("nan")


def eval_rf(kind, smiles, y, train_idx, test_idx, seed):
    X = fp_matrix(smiles, kind)
    rf = RandomForestRegressor(n_estimators=RF_TREES, n_jobs=-1, random_state=seed)
    rf.fit(X[train_idx], y[train_idx])
    return rf.predict(X[test_idx])


def eval_dmpnn(smiles, y, train_idx, test_idx, seed):
    import lightning as L
    import torch
    from chemprop import data, featurizers, models, nn
    from chemprop.data import BatchMolGraph

    torch.manual_seed(seed)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def dps(idx):
        return [data.MoleculeDatapoint.from_smi(smiles[i], np.array([y[i]])) for i in idx]

    train_dset = data.MoleculeDataset(dps(train_idx), feat)
    scaler = train_dset.normalize_targets()
    loader = data.build_dataloader(train_dset, batch_size=64, seed=seed)

    mp = nn.BondMessagePassing(depth=3)
    ffn = nn.RegressionFFN(
        n_tasks=1,
        input_dim=mp.output_dim,
        output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
    )
    model = models.MPNN(mp, nn.MeanAggregation(), ffn)
    accel = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=DMPNN_EPOCHS,
        accelerator=accel,
        devices=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, loader)
    model.eval()
    test_bmg = BatchMolGraph([feat(Chem.MolFromSmiles(smiles[i])) for i in test_idx])
    with torch.no_grad():
        pred = model(test_bmg).numpy(force=True).ravel()
    return pred


METHODS = ["D-MPNN", "Morgan+RF", "MACCS+RF", "RDKitTopo+RF", "AtomPair+RF"]


def run_endpoint(label, dataset):
    smiles, y, cliff = load_endpoint(dataset)
    logger.info(f"{label}: {len(smiles)} molecules, {int(cliff.sum())} cliff mols")
    per_fold = {m: [] for m in METHODS}
    per_fold_cliff = {m: [] for m in METHODS}

    for rep in range(N_REPEATS):
        folds = scaffold_folds(smiles, N_FOLDS, seed=rep)
        for fi in range(N_FOLDS):
            test_idx = np.array(folds[fi])
            train_idx = np.array([i for f in range(N_FOLDS) if f != fi for i in folds[f]])
            if len(test_idx) < 3 or len(train_idx) < 10:
                continue
            yt = y[test_idx]
            ct = cliff[test_idx]
            seed = rep * 100 + fi
            preds = {}
            preds["D-MPNN"] = eval_dmpnn(smiles, y, train_idx, test_idx, seed)
            preds["Morgan+RF"] = eval_rf("morgan", smiles, y, train_idx, test_idx, seed)
            preds["MACCS+RF"] = eval_rf("maccs", smiles, y, train_idx, test_idx, seed)
            preds["RDKitTopo+RF"] = eval_rf("rdkit_topo", smiles, y, train_idx, test_idx, seed)
            preds["AtomPair+RF"] = eval_rf("atom_pair", smiles, y, train_idx, test_idx, seed)
            for m in METHODS:
                per_fold[m].append(rmse(yt, preds[m]))
                if ct.sum() >= 2:
                    per_fold_cliff[m].append(rmse(yt[ct], preds[m][ct]))
            logger.info(
                f"  {label} rep{rep} fold{fi}: "
                + " ".join(f"{m}={per_fold[m][-1]:.3f}" for m in METHODS)
            )

    # Stats across methods on per-fold RMSE.
    arrays = [np.array(per_fold[m]) for m in METHODS]
    anova = f_oneway(*arrays)
    tuk = tukey_hsd(*arrays)
    tukey_p = tuk.pvalue.tolist()

    return {
        "label": label,
        "dataset": dataset,
        "n_molecules": len(smiles),
        "n_cliff": int(cliff.sum()),
        "methods": METHODS,
        "rmse_mean": {m: float(np.mean(per_fold[m])) for m in METHODS},
        "rmse_std": {m: float(np.std(per_fold[m])) for m in METHODS},
        "rmse_cliff_mean": {
            m: (float(np.mean(per_fold_cliff[m])) if per_fold_cliff[m] else None)
            for m in METHODS
        },
        "per_fold_rmse": {m: per_fold[m] for m in METHODS},
        "anova_F": float(anova.statistic),
        "anova_p": float(anova.pvalue),
        "tukey_pvalue_matrix": tukey_p,
    }


def summarize(results):
    lines = ["# CV + Tukey benchmark: learned vs. fixed fingerprints", ""]
    lines.append(
        f"{N_REPEATS}x{N_FOLDS}-fold scaffold CV, RMSE in pKi units "
        f"(lower is better). D-MPNN trained {DMPNN_EPOCHS} epochs on MPS; "
        f"fixed FPs fed to a {RF_TREES}-tree RandomForest."
    )
    lines.append("")
    for r in results:
        lines.append(f"## {r['label']}  (n={r['n_molecules']}, cliff={r['n_cliff']})")
        ranked = sorted(r["methods"], key=lambda m: r["rmse_mean"][m])
        for m in ranked:
            cliff = r["rmse_cliff_mean"][m]
            cliff_s = f", cliff RMSE {cliff:.3f}" if cliff is not None else ""
            lines.append(
                f"  - {m:14s} RMSE {r['rmse_mean'][m]:.3f} "
                f"+/- {r['rmse_std'][m]:.3f}{cliff_s}"
            )
        lines.append(
            f"  ANOVA F={r['anova_F']:.2f}, p={r['anova_p']:.2e}"
        )
        # D-MPNN vs each fixed FP significance
        di = r["methods"].index("D-MPNN")
        for j, m in enumerate(r["methods"]):
            if m == "D-MPNN":
                continue
            p = r["tukey_pvalue_matrix"][di][j]
            better = "better" if r["rmse_mean"]["D-MPNN"] < r["rmse_mean"][m] else "worse"
            sig = "significant" if p < 0.05 else "n.s."
            lines.append(f"    D-MPNN vs {m}: {better}, Tukey p={p:.3g} ({sig})")
        lines.append("")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for label, dataset in ENDPOINTS:
        results.append(run_endpoint(label, dataset))
        (OUT_DIR / "cv_tukey_results.json").write_text(json.dumps(results, indent=2))
    summary = summarize(results)
    (OUT_DIR / "cv_tukey_summary.md").write_text(summary)
    logger.info(f"done in {(time.time() - t0) / 60:.1f} min")
    print(summary)


if __name__ == "__main__":
    main()
