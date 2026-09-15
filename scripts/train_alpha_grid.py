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


def _project_2d(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    """PCA to 2D (cheap, deterministic) for the learned-fingerprint scatter."""
    x = emb - emb.mean(axis=0, keepdims=True)
    # SVD-based PCA; take first 2 components.
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:2].T).astype(np.float32)


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
        rep_proj = None
        rep_labels = None
        for seed in SEEDS:
            res = train_dmpnn(ed, alpha=alpha, epochs=EPOCHS, seed=seed)
            r2_a.append(res.r2_a)
            r2_b.append(res.r2_b)
            if seed == 0:
                rep_proj = _project_2d(res.embedding)
                rep_labels = ed.y  # (n, 2) pKi labels for coloring
            logger.info(
                f"  alpha={alpha} seed={seed}: "
                f"R2_a={res.r2_a:.3f} R2_b={res.r2_b:.3f}"
            )
        summary.append(
            {
                "alpha": alpha,
                "r2_a_mean": float(np.nanmean(r2_a)),
                "r2_a_std": float(np.nanstd(r2_a)),
                "r2_b_mean": float(np.nanmean(r2_b)),
                "r2_b_std": float(np.nanstd(r2_b)),
                # downsample the scatter to keep the JSON small
                "proj": rep_proj[::4].tolist() if rep_proj is not None else [],
                "labels": rep_labels[::4].tolist() if rep_labels is not None else [],
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
