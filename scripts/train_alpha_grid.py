"""Offline: pre-train the learned-fingerprint D-MPNN across a grid of alpha.

The notebook's climax is an interactive knob (alpha) that weights the training
loss between two endpoints. Training a D-MPNN takes ~a minute, far too slow to
run live per slider tick, so we train the whole grid here once and cache each
result (test R^2 on both endpoints + a 2D projection of the learned
fingerprint) as JSON. The notebook reads those and the slider just selects.

Runtime: ~1 min per (alpha, seed) on CPU; the default grid is a handful of
alphas x a few seeds per endpoint pair. Run it on whatever machine has time.

Usage:
    uv run python scripts/train_alpha_grid.py                 # all pairs
    uv run python scripts/train_alpha_grid.py --pair mu_vs_kappa
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from loguru import logger

from fingerprints.learned_fp import ENDPOINT_PAIRS, load_endpoint_data, train_dmpnn

warnings.filterwarnings("ignore")

OUT_DIR = Path("data/learned_fp")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)
EPOCHS = 30


def run_pair(pair_key: str) -> None:
    ed = load_endpoint_data(pair_key)
    logger.info(
        f"{pair_key}: {len(ed.smiles)} molecules "
        f"({ed.target_a} + {ed.target_b})"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-alpha: mean/std R^2 across seeds, plus one representative embedding
    # projection (from seed 0) for the scatter.
    summary: list[dict] = []
    for alpha in ALPHAS:
        r2_a, r2_b = [], []
        rmse_a, rmse_b = [], []
        scatter = None
        for seed in SEEDS:
            res = train_dmpnn(ed, alpha=alpha, epochs=EPOCHS, seed=seed)
            r2_a.append(res.r2_a)
            r2_b.append(res.r2_b)
            rmse_a.append(res.rmse_a)
            rmse_b.append(res.rmse_b)
            if seed == 0:
                # Full-population rows for the scatter: SMILES + both endpoints'
                # actual/pred/cliff + split tag, so the notebook can plot one
                # endpoint against the other and spotlight specific molecules
                # even when they fall in the train split (as cliff pairs do).
                def _col(arr, j):
                    return np.round(arr[:, j], 2).tolist()

                scatter = {
                    "smiles": list(res.smiles),
                    "actual_a": _col(res.all_actual, 0),
                    "actual_b": _col(res.all_actual, 1),
                    "pred_a": _col(res.all_pred, 0),
                    "pred_b": _col(res.all_pred, 1),
                    "cliff_a": res.all_cliff[:, 0].astype(int).tolist(),
                    "cliff_b": res.all_cliff[:, 1].astype(int).tolist(),
                    "is_test": res.is_test.astype(int).tolist(),
                    # Back-compat: keep the test-only per-endpoint shape too, so
                    # the RMSE fallback still works on either cache generation.
                    "a": {
                        "actual": np.round(res.test_actual[:, 0], 2).tolist(),
                        "pred": np.round(res.test_pred[:, 0], 2).tolist(),
                        "cliff": res.test_cliff[:, 0].astype(int).tolist(),
                    },
                    "b": {
                        "actual": np.round(res.test_actual[:, 1], 2).tolist(),
                        "pred": np.round(res.test_pred[:, 1], 2).tolist(),
                        "cliff": res.test_cliff[:, 1].astype(int).tolist(),
                    },
                }
            logger.info(
                f"  alpha={alpha} seed={seed}: "
                f"R2_a={res.r2_a:.3f} R2_b={res.r2_b:.3f} "
                f"RMSE_a={res.rmse_a:.3f} RMSE_b={res.rmse_b:.3f}"
            )
        summary.append(
            {
                "alpha": alpha,
                "r2_a_mean": float(np.nanmean(r2_a)),
                "r2_a_std": float(np.nanstd(r2_a)),
                "r2_b_mean": float(np.nanmean(r2_b)),
                "r2_b_std": float(np.nanstd(r2_b)),
                "rmse_a_mean": float(np.nanmean(rmse_a)),
                "rmse_a_std": float(np.nanstd(rmse_a)),
                "rmse_b_mean": float(np.nanmean(rmse_b)),
                "rmse_b_std": float(np.nanstd(rmse_b)),
                "scatter": scatter,
            }
        )

    out = {
        "pair_key": pair_key,
        "target_a": ed.target_a,
        "target_b": ed.target_b,
        "n_molecules": len(ed.smiles),
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "results": summary,
    }
    path = OUT_DIR / f"{pair_key}_alpha_grid.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None, help="one endpoint-pair key, or all")
    args = ap.parse_args()
    pairs = [args.pair] if args.pair else list(ENDPOINT_PAIRS)
    for pk in pairs:
        run_pair(pk)


if __name__ == "__main__":
    main()
