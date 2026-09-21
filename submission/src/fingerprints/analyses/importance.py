"""Offline: train RandomForest regressors on Dopamine D3 and D4 activity for two
fingerprints - ECFP (Morgan) and the pretrained CheMeleon embedding - and cache
each model's per-bit feature importances (+ held-out R2/RMSE).

The notebook reads these cached importances to:
  * color the cliff-pair molecules by aggregated feature importance
  * tint the fingerprint strips by per-bit importance

Fingerprints are frozen inputs; only a small RF is trained. CheMeleon featurizing
is the only heavy step, so we cache everything here and the notebook stays instant.

Run:
  uv run python scripts/train_importance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as fpg
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from fingerprints import chemeleon_fp as chf

RDLogger.DisableLog("rdApp.*")

from fingerprints import paths
from fingerprints.data import molace

CACHE_MOLACE = paths.CACHE_DIR / "molace"
OUT_DIR = paths.IMPORTANCE.parent
N_BITS = 2048
RF_TREES = 400

ENDPOINTS = [
    ("Dopamine D3", "CHEMBL234_Ki"),
    ("Dopamine D4", "CHEMBL219_Ki"),
    ("mu-opioid", "CHEMBL233_Ki"),
    ("kappa-opioid", "CHEMBL237_Ki"),
]

_MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=N_BITS)


def load_endpoint(dataset: str):
    df = pl.read_csv(CACHE_MOLACE / f"{dataset}.csv")
    smis, ys = [], []
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
    return smis, np.array(ys)


def ecfp_matrix(smiles):
    return np.asarray(
        [_MORGAN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
        dtype=np.float32,
    )


def chemeleon_matrix(smiles):
    rows = []
    for i, s in enumerate(smiles):
        rows.append(chf.fingerprint(Chem.MolFromSmiles(s)))
        if (i + 1) % 200 == 0:
            logger.info(f"    CheMeleon featurized {i + 1}/{len(smiles)}")
    return np.asarray(rows, dtype=np.float32)


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(yt, yp):
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def train_one(X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    rf = RandomForestRegressor(n_estimators=RF_TREES, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    yp = rf.predict(Xte)
    return {
        "importances": rf.feature_importances_.astype(float).tolist(),
        "r2": r2(yte, yp),
        "rmse": rmse(yte, yp),
        "n_train": len(ytr),
        "n_test": len(yte),
    }


def _ensure_molace(dataset: str):
    ds = molace.MolACEDataset(
        name=dataset, target_label=dataset, target_class="", assay_type="Ki"
    )
    molace.download_molace(ds, CACHE_MOLACE / f"{dataset}.csv")


def main(out_dir=None, on_step=None):
    """Train the per-fingerprint feature-importance models.

    Args:
        out_dir: where to write ``rf_importances.json`` (defaults to OUT_DIR).
        on_step: optional ``(label)`` callback for a progress bar.
    """
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"n_bits": N_BITS, "rf_trees": RF_TREES, "endpoints": {}}
    for label, dataset in ENDPOINTS:
        if on_step is not None:
            on_step(label)
        _ensure_molace(dataset)
        logger.info(f"{label}: loading {dataset}")
        smiles, y = load_endpoint(dataset)
        logger.info(f"  {len(smiles)} molecules")
        logger.info("  building ECFP matrix")
        Xe = ecfp_matrix(smiles)
        logger.info("  building CheMeleon matrix (forward passes)")
        Xc = chemeleon_matrix(smiles)
        logger.info("  training RFs")
        ecfp = train_one(Xe, y)
        chem = train_one(Xc, y)
        logger.info(
            f"  ECFP: R2={ecfp['r2']:.3f} RMSE={ecfp['rmse']:.3f} | "
            f"CheMeleon: R2={chem['r2']:.3f} RMSE={chem['rmse']:.3f}"
        )
        out["endpoints"][label] = {"ecfp": ecfp, "chemeleon": chem}
    path = out_dir / "rf_importances.json"
    path.write_text(json.dumps(out))
    logger.info(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def endpoint_labels() -> list[str]:
    return [lbl for lbl, _ in ENDPOINTS]


if __name__ == "__main__":
    main()
