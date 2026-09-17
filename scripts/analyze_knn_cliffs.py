"""Offline: analyse *why a similarity model can't see activity cliffs*, using the
simplest model whose entire behaviour IS the fingerprint's notion of similarity:
k-nearest-neighbours regression on ECFP (Tanimoto).

For each endpoint (Dopamine D3, D4) we cache:

  * ``k_curve``: held-out R^2 as a function of k - the real bias/variance
    tradeoff (memorise locally at small k vs. smooth globally at large k). This
    is the honest analog of "move the decision boundary": the *only* knob kNN
    has is neighbourhood size, and it genuinely trades local sensitivity for
    global accuracy.

  * per curated cliff pair (that lives in this endpoint's data):
      - each molecule's nearest neighbours (smiles, Tanimoto, activity) from the
        rest of the dataset - so the notebook can SHOW that the two cliff
        molecules sit in the same fingerprint neighbourhood;
      - each molecule's leave-one-out kNN-predicted activity across a sweep of k,
        alongside its true activity - so the notebook can show the prediction
        tracks the neighbourhood (right for one end of the cliff, wrong for the
        other) no matter what k is chosen.

Why kNN and not a tree / MLP: kNN's prediction is literally the average activity
of the fingerprint-nearest molecules, so the fingerprint's similarity function
*is* the model - nothing is learned on top. Its blindness to the cliff is
therefore the fingerprint's blindness, laid bare, not an artefact of a fancier
learner.

Run:
  uv run python scripts/analyze_knn_cliffs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as fpg

from fingerprints.data import context_cliffs as ctx

RDLogger.DisableLog("rdApp.*")

CACHE_MOLACE = Path(".cache/molace")
OUT_DIR = Path("data/knn_cliffs")
N_BITS = 2048
_MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=N_BITS)

# endpoint label -> molace dataset file (same mapping the RF importance uses)
ENDPOINTS = {
    "Dopamine D3": "CHEMBL234_Ki",
    "Dopamine D4": "CHEMBL219_Ki",
}
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100]
N_NEIGHBORS = 4  # neighbours to record per cliff molecule
SIM_THRESHOLD = 0.7  # "structurally similar" cutoff for the smoothness stat
FLAT_GAP = 1.0  # |dpKi| below this = a flat (smooth) pair
CLIFF_GAP = 2.0  # |dpKi| above this = an activity cliff


def load_endpoint(dataset: str):
    """canonical smiles -> (activity, split); de-duplicated by canonical SMILES."""
    df = pl.read_csv(CACHE_MOLACE / f"{dataset}.csv")
    smis, ys, splits = [], [], []
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
        splits.append(row["split"])
    return smis, np.array(ys), np.array(splits)


def ecfp_matrix(smiles):
    return np.asarray(
        [_MORGAN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
        dtype=np.float32,
    )


def tanimoto(A, B):
    """(n,d) x (m,d) binary -> (n,m) Tanimoto similarity."""
    inter = A @ B.T
    a = A.sum(1)[:, None]
    b = B.sum(1)[None, :]
    union = a + b - inter
    return np.where(union > 0, inter / union, 0.0)


def r2(yt, yp):
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def knn_predict(sim_to_ref, y_ref, k):
    """Mean activity of the top-k most-similar reference molecules, per query."""
    idx = np.argsort(-sim_to_ref, axis=1)[:, :k]
    return y_ref[idx].mean(1)


def analyse_endpoint(label: str, dataset: str) -> dict:
    smis, y, split = load_endpoint(dataset)
    X = ecfp_matrix(smis)
    idx_of = {s: i for i, s in enumerate(smis)}
    tr = split == "train"
    te = split == "test"
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # --- smoothness census: among structurally similar pairs, how many are
    #     flat vs cliffs? This is the load-bearing evidence for *why any*
    #     structure-only model must default to 'similar structure -> similar
    #     activity': that assumption holds for the vast majority of similar
    #     pairs, so honouring the rare cliff would wreck accuracy everywhere. ---
    S_all = tanimoto(X, X)
    n = len(smis)
    iu = np.triu_indices(n, 1)
    tt = S_all[iu]
    dd = np.abs(y[iu[0]] - y[iu[1]])
    sim_mask = tt >= SIM_THRESHOLD
    n_similar = int(sim_mask.sum())
    dd_sim = dd[sim_mask]
    # histogram of activity gaps among similar pairs (for the visual)
    gap_edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0]
    gap_hist = []
    for lo, hi in zip(gap_edges[:-1], gap_edges[1:]):
        c = int(np.sum((dd_sim >= lo) & (dd_sim < hi)))
        gap_hist.append({"lo": lo, "hi": hi, "count": c})
    smoothness = {
        "sim_threshold": SIM_THRESHOLD,
        "n_similar_pairs": n_similar,
        "frac_flat": float(np.mean(dd_sim < FLAT_GAP)) if n_similar else 0.0,
        "frac_cliff": float(np.mean(dd_sim > CLIFF_GAP)) if n_similar else 0.0,
        "flat_gap": FLAT_GAP,
        "cliff_gap": CLIFF_GAP,
        "gap_hist": gap_hist,
    }

    # --- held-out R^2 vs k (test molecules predicted from train neighbours) ---
    S_te_tr = tanimoto(Xte, Xtr)
    k_curve = []
    for k in K_GRID:
        kk = min(k, len(ytr))
        yp = knn_predict(S_te_tr, ytr, kk)
        k_curve.append({"k": k, "r2": r2(yte, yp)})

    # --- per cliff pair: neighbours + predicted-vs-true across k ---
    # Reference set for a cliff molecule = the whole dataset minus itself (a
    # leave-one-out neighbourhood). This shows where the molecule actually sits.
    tp = ctx.by_key()["D3_vs_D4"]
    pairs_out = []
    for i, cl in enumerate(tp.cliffs):
        c1 = Chem.MolToSmiles(Chem.MolFromSmiles(cl.smiles_1))
        c2 = Chem.MolToSmiles(Chem.MolFromSmiles(cl.smiles_2))
        if c1 not in idx_of or c2 not in idx_of:
            continue  # this endpoint doesn't contain the pair

        def molecule_report(self_smi):
            qi = idx_of[self_smi]
            sims = tanimoto(X[qi : qi + 1], X)[0]
            sims[qi] = -1.0  # exclude self
            order = np.argsort(-sims)
            neighbors = [
                {
                    "smiles": smis[j],
                    "tanimoto": float(sims[j]),
                    "activity": float(y[j]),
                }
                for j in order[:N_NEIGHBORS]
            ]
            # kNN prediction across the grid (leave-one-out neighbourhood).
            pred_by_k = []
            for k in K_GRID:
                kk = min(k, len(order))
                top = order[:kk]
                pred_by_k.append({"k": k, "pred": float(y[top].mean())})
            return {
                "smiles": self_smi,
                "true": float(y[qi]),
                "neighbors": neighbors,
                "pred_by_k": pred_by_k,
            }

        pairs_out.append(
            {
                "index": i,
                "cliff_on": cl.cliff_on,
                "change": cl.change,
                "mol1": molecule_report(c1),
                "mol2": molecule_report(c2),
            }
        )

    return {
        "n_total": int(len(smis)),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "smoothness": smoothness,
        "k_curve": k_curve,
        "cliff_pairs": pairs_out,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"n_bits": N_BITS, "k_grid": K_GRID, "endpoints": {}}
    for label, dataset in ENDPOINTS.items():
        logger.info(f"{label}: analysing {dataset}")
        res = analyse_endpoint(label, dataset)
        best = max(res["k_curve"], key=lambda d: d["r2"])
        logger.info(
            f"  n={res['n_total']} | best k={best['k']} R2={best['r2']:.3f} | "
            f"{len(res['cliff_pairs'])} cliff pairs present"
        )
        out["endpoints"][label] = res
    path = OUT_DIR / "knn_cliffs.json"
    path.write_text(json.dumps(out))
    logger.info(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
