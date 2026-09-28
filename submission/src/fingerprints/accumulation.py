"""Can a model *accumulate* a whole-molecule quantity from a fingerprint?

The activity-cliff census argues that solubility-type cliffs arise because a
fingerprint can't track a property that **accumulates over the whole molecule**
(add one more -CH2- and solubility drops, but a binary fingerprint barely
moves). This module turns that claim into a controlled, runs-live experiment
across several fingerprint encodings.

Encodings + heads compared (all with a fairly-tuned linear head via RidgeCV,
plus one deep MLP on the binary fingerprint to ask whether depth can rescue it):

  * **binary Morgan** — the usual fingerprint. Binarising throws away *how many
    times* each substructure occurs, i.e. exactly the count information an
    accumulator needs.
  * **count Morgan** — same bits, but each slot holds a count, so a linear head
    can literally sum them.
  * **binary Morgan + deep MLP** — can depth reconstruct the lost counts from
    which bits co-occur?
  * **CheMeleon (learned)** — a pretrained message-passing fingerprint. It is
    *mean*-pooled over atoms, which is telling: a mean is size-*intensive*, so
    even a learned representation is not automatically good at a size-extensive
    raw count.

Target 1 is a **pure accumulator** we control exactly (heavy-atom count — by
definition a sum over atoms). Target 2 is **real AqSolDB solubility**. The pure
target isolates the mechanism; the real one shows how much carries over.

Everything trains live on a Bemis-Murcko scaffold split (no leakage). CheMeleon
features are batched so the whole thing stays to a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

from fingerprints import paths
from fingerprints.data import tdc

RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
_GEN = fpg.GetMorganGenerator(radius=2, fpSize=N_BITS)
_SALT = SaltRemover()
_MAX_N = 4000  # cap for a snappy live fit; sampled deterministically
_ALPHAS = np.logspace(-2, 4, 13)  # RidgeCV search grid (headline section)
_SURVEY_MAX_N = 2500  # smaller cap for the multi-fingerprint survey (speed)
_SURVEY_ALPHAS = np.array([1.0, 10.0, 100.0, 1000.0])  # coarse but fair grid

# The classical RDKit fingerprints the survey compares, each in binary and
# count form. All fold to N_BITS bits so the only thing that changes between
# "binary" and "count" is whether a slot records presence or multiplicity.
_SURVEY_GENERATORS = {
    "Morgan (ECFP4)": fpg.GetMorganGenerator(radius=2, fpSize=N_BITS),
    "RDKit topological": fpg.GetRDKitFPGenerator(fpSize=N_BITS),
    "atom-pair": fpg.GetAtomPairGenerator(fpSize=N_BITS),
    "topological torsion": fpg.GetTopologicalTorsionGenerator(fpSize=N_BITS),
}

# One binding endpoint (MoleculeACE / ChEMBL) to contrast against solubility:
# does counting help or hurt when the property is molecular *recognition*
# rather than an accumulated bulk quantity?
_BINDING_ENDPOINT = ("Dopamine D3 (pKi)", "CHEMBL234_Ki")


def _parent(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _SALT.StripMol(mol, dontRemoveEverything=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=1)
def _dataset():
    """Load, desalt, dedup AqSolDB → (mols, smiles, logS). Cached per session."""
    df = tdc.load_tdc(tdc.SOLUBILITY, paths.CACHE_DIR / "tdc")
    by_parent: dict[str, list[float]] = {}
    for row in df.iter_rows(named=True):
        p = _parent(row["smiles"])
        if p is None or row["y"] is None:
            continue
        by_parent.setdefault(p, []).append(float(row["y"]))
    smis = list(by_parent)
    ys = np.array([float(np.mean(by_parent[s])) for s in smis])
    if len(smis) > _MAX_N:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(smis), _MAX_N, replace=False)
        smis = [smis[i] for i in idx]
        ys = ys[idx]
    mols = [Chem.MolFromSmiles(s) for s in smis]
    return mols, smis, ys


@lru_cache(maxsize=1)
def _morgan_matrices():
    """(binary fp, count fp) Morgan matrices for the cached dataset."""
    mols, _, _ = _dataset()
    xb = np.vstack(
        [np.asarray(_GEN.GetFingerprintAsNumPy(m), dtype=np.float32) for m in mols]
    )
    xc = np.vstack(
        [np.asarray(_GEN.GetCountFingerprintAsNumPy(m), dtype=np.float32) for m in mols]
    )
    return xb, xc


@lru_cache(maxsize=1)
def _chemeleon_matrix():
    """Batched mean-pooled CheMeleon fingerprints for the cached dataset."""
    from fingerprints import chemeleon_fp as chf

    mols, _, _ = _dataset()
    return chf.fingerprint_matrix(mols)


@lru_cache(maxsize=1)
def _scaffold_split():
    """Boolean (train, test) masks by Bemis-Murcko scaffold (no leakage)."""
    mols, _smis, _ = _dataset()
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(mols):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            sc = ""
        groups.setdefault(sc or f"__none_{i}", []).append(i)
    n = len(mols)
    is_test = np.zeros(n, dtype=bool)
    target = int(round(0.2 * n))
    for grp in sorted(groups.values(), key=len):
        if is_test.sum() >= target:
            break
        for i in grp:
            is_test[i] = True
    return ~is_test, is_test


def _targets():
    """The two regression targets: a pure accumulator + real solubility."""
    mols, _, ys = _dataset()
    hac = np.array([m.GetNumHeavyAtoms() for m in mols], dtype=float)
    return {
        "heavy-atom count": {
            "y": hac,
            "kind": "pure accumulator (Σ over atoms)",
            "unit": "atoms",
        },
        "aqueous solubility": {
            "y": ys,
            "kind": "real measured property",
            "unit": "logS",
        },
    }


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


@dataclass(frozen=True)
class ModelScore:
    key: str
    label: str
    family: str  # "binary" | "count" | "learned" — drives the chart colour
    r2: float


def target_labels() -> list[str]:
    return list(_targets().keys())


def n_molecules() -> int:
    if has_data():
        return _cache()["n_molecules"]
    return len(_dataset()[0])


def target_meta(target: str) -> dict:
    return {k: v for k, v in _targets()[target].items() if k != "y"}


def _ridge_r2(x, y, tr, te, *, scale: bool) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [StandardScaler()] if scale else []
    steps.append(RidgeCV(alphas=_ALPHAS))
    model = make_pipeline(*steps).fit(x[tr], y[tr])
    return _r2(y[te], model.predict(x[te]))


def _mlp_r2(x, y, tr, te, *, depth: int, width: int) -> float:
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(
        hidden_layer_sizes=(width,) * depth,
        max_iter=300,
        random_state=0,
        early_stopping=True,
    ).fit(x[tr], y[tr])
    return _r2(y[te], model.predict(x[te]))


@lru_cache(maxsize=8)
def scores(target: str, depth: int = 4, width: int = 48) -> tuple[ModelScore, ...]:
    """Held-out R² for every encoding/head on ``target``.

    Reads the shipped cache under ``data/`` when present (instant); otherwise
    fits live. Regenerate with ``python -m fingerprints.accumulation``.
    """
    if has_data():
        return tuple(
            ModelScore(**d) for d in _cache()["scores"].get(target, [])
        )
    return _scores_live(target, depth=depth, width=width)


def _scores_live(
    target: str, depth: int = 4, width: int = 48
) -> tuple[ModelScore, ...]:
    """Fit every encoding/head on ``target`` and return held-out R² for each.

    Linear heads use RidgeCV (fair, auto-tuned regularisation); CheMeleon's
    embedding is standardised first because a few of its dimensions have very
    large scale. The binary-Morgan MLP is the "can depth rescue it?" probe.
    """
    xb, xc = _morgan_matrices()
    tr, te = _scaffold_split()
    y = _targets()[target]["y"]

    out = [
        ModelScore(
            "binary_linear", "binary Morgan + linear", "binary",
            _ridge_r2(xb, y, tr, te, scale=False),
        ),
        ModelScore(
            "count_linear", "count Morgan + linear", "count",
            _ridge_r2(xc, y, tr, te, scale=False),
        ),
        ModelScore(
            "binary_mlp", f"binary Morgan + deep MLP ({width}×{depth})", "binary",
            _mlp_r2(xb, y, tr, te, depth=depth, width=width),
        ),
    ]
    try:
        xche = _chemeleon_matrix()
        out.append(
            ModelScore(
                "chemeleon_linear", "CheMeleon (learned) + linear", "learned",
                _ridge_r2(xche, y, tr, te, scale=True),
            )
        )
    except Exception:
        # CheMeleon weights unavailable — skip it rather than fail the cell.
        pass
    return tuple(out)


# ---------------------------------------------------------------------------
# Follow-up survey: binary vs count across every classical RDKit fingerprint,
# on a pure accumulator, on solubility, and on a binding endpoint.
# ---------------------------------------------------------------------------


def _scaffold_masks(mols, test_frac: float = 0.2):
    """Generic Bemis-Murcko scaffold split for an arbitrary molecule list."""
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(mols):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            sc = ""
        groups.setdefault(sc or f"__none_{i}", []).append(i)
    n = len(mols)
    is_test = np.zeros(n, dtype=bool)
    target = int(round(test_frac * n))
    for grp in sorted(groups.values(), key=len):
        if is_test.sum() >= target:
            break
        for i in grp:
            is_test[i] = True
    return ~is_test, is_test


@lru_cache(maxsize=1)
def _binding_dataset():
    """Load a MoleculeACE binding endpoint into (mols, pKi). Cached per session.

    Downloads the CSV via the MoleculeACE loader if not already present, so the
    survey is self-contained on a cold clone.
    """
    import polars as pl

    from fingerprints.data import molace

    _label, name = _BINDING_ENDPOINT
    path = paths.CACHE_DIR / "molace" / f"{name}.csv"
    if not path.exists():
        ds = molace.MolACEDataset(
            name=name, target_label=name, target_class="", assay_type="Ki"
        )
        molace.download_molace(ds, path)
    df = pl.read_csv(path)
    smis, ys, seen = [], [], set()
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
    mols = [Chem.MolFromSmiles(s) for s in smis]
    if len(mols) > _SURVEY_MAX_N:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(mols), _SURVEY_MAX_N, replace=False)
        mols = [mols[i] for i in idx]
        ys = [ys[i] for i in idx]
    return mols, np.asarray(ys, dtype=float)


def _fp_matrix(mols, gen, *, count: bool) -> np.ndarray:
    fn = gen.GetCountFingerprintAsNumPy if count else gen.GetFingerprintAsNumPy
    return np.vstack([np.asarray(fn(m), dtype=np.float32) for m in mols])


def _survey_ridge_r2(x, y, tr, te) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(with_mean=False), RidgeCV(alphas=_SURVEY_ALPHAS)
    ).fit(x[tr], y[tr])
    return _r2(y[te], model.predict(x[te]))


@dataclass(frozen=True)
class SurveyCell:
    fingerprint: str
    encoding: str  # "binary" or "count"
    target: str
    r2: float


def survey_targets() -> list[str]:
    """Column labels for the survey, in display order."""
    return ["heavy-atom count", "aqueous solubility", _BINDING_ENDPOINT[0]]


def survey_fingerprints() -> list[str]:
    return list(_SURVEY_GENERATORS)


def survey_n() -> dict[str, int]:
    """Molecule counts backing each target family."""
    if has_data():
        return _cache()["survey_n"]
    return {
        "accumulator": len(_dataset()[0]),
        "binding": len(_binding_dataset()[0]),
    }


@lru_cache(maxsize=1)
def survey() -> tuple[SurveyCell, ...]:
    """binary vs count for every classical fingerprint on all survey targets.

    Reads the shipped cache under ``data/`` when present (instant); otherwise
    fits live. Regenerate with ``python -m fingerprints.accumulation``.
    """
    if has_data():
        return tuple(SurveyCell(**d) for d in _cache()["survey"])
    return _survey_live()


def _survey_live() -> tuple[SurveyCell, ...]:
    """binary vs count for every classical fingerprint on all survey targets.

    ADMET targets (heavy-atom count, solubility) share the AqSolDB molecule set
    and its scaffold split; the binding target uses its own MoleculeACE set and
    split. Every fit is a fairly-tuned RidgeCV on standardised features.
    """
    mols_s, _, ys_s = _dataset()
    tr_s, te_s = _scaffold_split()
    hac = np.array([m.GetNumHeavyAtoms() for m in mols_s], dtype=float)
    mols_b, ys_b = _binding_dataset()
    tr_b, te_b = _scaffold_masks(mols_b)
    bind_label = _BINDING_ENDPOINT[0]

    admet_targets = [("heavy-atom count", hac), ("aqueous solubility", ys_s)]
    out: list[SurveyCell] = []
    for fp_label, gen in _SURVEY_GENERATORS.items():
        for encoding, count in [("binary", False), ("count", True)]:
            xs = _fp_matrix(mols_s, gen, count=count)
            for tgt, y in admet_targets:
                out.append(
                    SurveyCell(
                        fp_label, encoding, tgt, _survey_ridge_r2(xs, y, tr_s, te_s)
                    )
                )
            xb = _fp_matrix(mols_b, gen, count=count)
            out.append(
                SurveyCell(
                    fp_label,
                    encoding,
                    bind_label,
                    _survey_ridge_r2(xb, ys_b, tr_b, te_b),
                )
            )
    return tuple(out)


def count_minus_binary(target: str) -> list[tuple[str, float]]:
    """Per-fingerprint count-minus-binary R2 delta on ``target``.

    Positive means counting helps; negative means counting hurts.
    """
    by = {(c.fingerprint, c.encoding): c.r2 for c in survey() if c.target == target}
    deltas = []
    for fp in survey_fingerprints():
        b = by.get((fp, "binary"))
        c = by.get((fp, "count"))
        if b is not None and c is not None:
            deltas.append((fp, c - b))
    return deltas


# ---------------------------------------------------------------------------
# Disk cache: all live fits (scores over every target + the survey grid) are
# precomputed and shipped under data/ so the notebook loads instantly. Rebuild
# with ``python -m fingerprints.accumulation``.
# ---------------------------------------------------------------------------


def has_data() -> bool:
    return paths.ACCUMULATION.exists()


@lru_cache(maxsize=1)
def _cache() -> dict:
    import json

    return json.loads(paths.ACCUMULATION.read_text())


def main() -> None:
    """Fit everything live and write the JSON cache read by :func:`scores`,
    :func:`survey`, :func:`n_molecules`, and :func:`survey_n`."""
    import json
    from dataclasses import asdict

    payload = {
        "n_molecules": len(_dataset()[0]),
        "survey_n": {
            "accumulator": len(_dataset()[0]),
            "binding": len(_binding_dataset()[0]),
        },
        "scores": {
            target: [asdict(s) for s in _scores_live(target)]
            for target in target_labels()
        },
        "survey": [asdict(c) for c in _survey_live()],
    }
    paths.ACCUMULATION.parent.mkdir(parents=True, exist_ok=True)
    paths.ACCUMULATION.write_text(json.dumps(payload, indent=2))
    n = sum(len(v) for v in payload["scores"].values()) + len(payload["survey"])
    print(f"wrote {n} fits -> {paths.ACCUMULATION}")


if __name__ == "__main__":
    main()
