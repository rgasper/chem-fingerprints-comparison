"""Aggregate cliff-vs-non-cliff figures using matched controls and PR-AUC.

Builds three figures:

- 04_pr_auc_summary.png: PR-AUC heatmap per (target, FP), same layout
  as the existing 03_cliff_blind_summary.png. Higher = better separation
  of cliff pairs from matched non-cliff controls.
- 05_metric_scatter.png: per-target scatter of cliff-blind-rate (x) vs
  PR-AUC (y). Shows how the absolute-threshold and rank-based metrics
  agree or disagree per (target, FP).
- 06_neural_metric_compare.png: for the two neural FPs, paired cliff
  vs matched-non-cliff distributions under cosine and L2, one column
  per target. Tests whether neural cliff-blindness is a metric artifact.

Reuses the cliff-pair cache from `figure_cliffs.py`. Caches matched
non-cliff sets at `.cache/noncliff_pairs/<dataset>.pkl` since the MCS
verification is also expensive.

Run:
  uv run python scripts/figure_cliffs_aggregate.py
  uv run python scripts/figure_cliffs_aggregate.py --no-cache  # recompute
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.cliffs import CliffPair, find_cliff_pairs
from fingerprints.clustering.cliffs_aggregate import (
    CliffSeparationResult,
    NonCliffPair,
    cliff_separation_for_all,
    sample_matched_noncliffs,
)
from fingerprints.data.molace import (
    D3_DOPAMINE,
    GSK3B,
    THROMBIN,
    MolACEDataset,
    load_molace,
)
from fingerprints.fingerprint_methods import rdkit_fps
from fingerprints.fingerprint_methods.base import FingerprintResult
from fingerprints.fingerprint_methods.chemeleon_fp import CheMeleonFingerprint
from fingerprints.fingerprint_methods.mist_fp import MIST_28M, MISTFingerprint
from fingerprints.plots.cliffs_aggregate import (
    plot_metric_scatter,
    plot_neural_metric_compare,
    plot_pr_auc_summary,
)


CACHE_MOLACE = Path(".cache/molace")
CACHE_CLIFFS = Path(".cache/cliff_pairs")
CACHE_NONCLIFFS = Path(".cache/noncliff_pairs")
FIG_DIR = Path("figures/cliffs")

DATASETS = (D3_DOPAMINE, THROMBIN, GSK3B)
NEURAL_SHORT_IDS = ("chemeleon", "mist_28M")


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _build_fps(
    mols: list,
    chemeleon: CheMeleonFingerprint,
    mist_28m: MISTFingerprint,
) -> dict[str, FingerprintResult]:
    fps: dict[str, FingerprintResult] = dict(rdkit_fps.all_classical(mols))
    fps["chemeleon"] = chemeleon(mols)
    fps["mist_28M"] = mist_28m(mols)
    return fps


def _load_dataset(ds: MolACEDataset) -> tuple[list, np.ndarray]:
    """Return (mols, y) with rows where SMILES failed dropped."""
    df = load_molace(ds, CACHE_MOLACE)
    smis = df["smiles"].to_list()
    parsed = [(MolFromSmiles(s), i) for i, s in enumerate(smis)]
    keep = [(m, i) for m, i in parsed if m is not None]
    mols = [m for m, _ in keep]
    keep_idx = np.array([i for _, i in keep])
    y = df["y"].to_numpy()[keep_idx]
    if len(mols) < df.height:
        logger.info(f"  parsed {len(mols)}/{df.height} mols")
    return mols, y


def _load_or_compute_cliffs(
    ds: MolACEDataset,
    mols: list,
    y: np.ndarray,
    use_cache: bool = True,
) -> list[CliffPair]:
    cache_path = CACHE_CLIFFS / f"{ds.name}.pkl"
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            pairs = pickle.load(fh)
        logger.info(f"loaded {len(pairs)} cliff pairs from {cache_path}")
        return pairs
    pairs = find_cliff_pairs(mols, y)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(pairs, fh)
    logger.info(f"cached {len(pairs)} cliff pairs to {cache_path}")
    return pairs


def _load_or_sample_noncliffs(
    ds: MolACEDataset,
    mols: list,
    y: np.ndarray,
    cliff_pairs: list[CliffPair],
    use_cache: bool = True,
) -> list[NonCliffPair]:
    cache_path = CACHE_NONCLIFFS / f"{ds.name}.pkl"
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            pairs = pickle.load(fh)
        logger.info(f"loaded {len(pairs)} matched non-cliffs from {cache_path}")
        return pairs
    pairs = sample_matched_noncliffs(mols, y, cliff_pairs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(pairs, fh)
    logger.info(f"cached {len(pairs)} matched non-cliffs to {cache_path}")
    return pairs


def main(use_cache: bool = True) -> None:
    device = _device()
    logger.info(f"using device={device}")

    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    cosine_results: dict[str, dict[str, CliffSeparationResult]] = {}
    l2_results: dict[str, dict[str, CliffSeparationResult]] = {}

    for ds in DATASETS:
        logger.info(f"=== {ds.name} ({ds.target_label}) ===")
        mols, y = _load_dataset(ds)
        cliff_pairs = _load_or_compute_cliffs(ds, mols, y, use_cache=use_cache)
        if not cliff_pairs:
            logger.warning(f"{ds.name}: no cliff pairs; skipping")
            continue

        noncliff_pairs = _load_or_sample_noncliffs(
            ds, mols, y, cliff_pairs, use_cache=use_cache,
        )
        if not noncliff_pairs:
            logger.warning(
                f"{ds.name}: no matched non-cliffs sampled; skipping"
            )
            continue

        fps = _build_fps(mols, chemeleon, mist_28m)

        # Default-metric results (jaccard for binary, cosine for neural)
        sep = cliff_separation_for_all(fps, cliff_pairs, noncliff_pairs)
        cosine_results[ds.target_label] = sep

        # L2 override for the two neural FPs only
        l2_sep = cliff_separation_for_all(
            {k: v for k, v in fps.items() if k in NEURAL_SHORT_IDS},
            cliff_pairs,
            noncliff_pairs,
            extra_metrics={k: "euclidean" for k in NEURAL_SHORT_IDS},
        )
        l2_results[ds.target_label] = l2_sep

        del fps

    if not cosine_results:
        logger.error("no datasets produced results; exiting")
        return

    # Figure 04: PR-AUC heatmap
    plot_pr_auc_summary(
        cosine_results,
        out_path=FIG_DIR / "04_pr_auc_summary.png",
        title=(
            "PR-AUC: cliff vs matched-non-cliff separation per fingerprint "
            "(default metric)"
        ),
    )

    # Figure 05: scatter of cliff-blind-rate vs PR-AUC
    cliff_blind_by_dataset: dict[str, dict[str, float]] = {
        ds: {sid: r.cliff_blind_rate_07 for sid, r in res.items()}
        for ds, res in cosine_results.items()
    }
    pr_auc_by_dataset: dict[str, dict[str, float]] = {
        ds: {sid: r.pr_auc for sid, r in res.items()}
        for ds, res in cosine_results.items()
    }
    plot_metric_scatter(
        cliff_blind_by_dataset,
        pr_auc_by_dataset,
        out_path=FIG_DIR / "05_metric_scatter.png",
        title=(
            "Absolute threshold (x) vs rank-based PR-AUC (y) per fingerprint"
        ),
    )

    # Figure 06: neural FP cosine vs L2
    plot_neural_metric_compare(
        cosine_results_by_dataset=cosine_results,
        l2_results_by_dataset=l2_results,
        neural_short_ids=list(NEURAL_SHORT_IDS),
        out_path=FIG_DIR / "06_neural_metric_compare.png",
        title=(
            "Neural fingerprints: cliff vs matched-non-cliff "
            "under cosine and L2"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore cached cliff candidates and recompute",
    )
    args = parser.parse_args()
    main(use_cache=not args.no_cache)
