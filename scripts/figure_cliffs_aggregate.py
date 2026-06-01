"""Aggregate cliff-vs-non-cliff figures using matched controls and PR-AUC.

Two layers:

1. Headline 3-target detail (D3 dopamine, Thrombin, GSK-3 beta):
   - 04_pr_auc_summary.png: PR-AUC heatmap, annotated with bootstrap CIs.
   - 05_metric_scatter.png: per-target scatter cliff-blind-rate vs PR-AUC,
     with bootstrap CI error bars.
   - 06_neural_metric_compare.png: paired cliff vs non-cliff distributions
     under cosine and L2 for the two neural FPs.

2. Cross-target sweep (all 30 MoleculeACE targets, dropping any that produce
   fewer than MIN_CLIFFS cliffs to keep stats meaningful):
   - 07_pr_auc_cross_target_boxplot.png: per-FP boxplot of PR-AUC across
     targets. Strip plot overlay shows individual targets.

Caches per-target cliff-pairs and matched non-cliffs so repeat runs are fast.

Run:
  uv run python scripts/figure_cliffs_aggregate.py
  uv run python scripts/figure_cliffs_aggregate.py --no-cache  # recompute
  uv run python scripts/figure_cliffs_aggregate.py --skip-sweep  # 3-target only
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
    ALL_MOLACE_DATASETS,
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
    plot_cross_target_boxplot,
    plot_metric_scatter,
    plot_neural_metric_compare,
    plot_pr_auc_summary,
)


CACHE_MOLACE = Path(".cache/molace")
CACHE_CLIFFS = Path(".cache/cliff_pairs")
CACHE_NONCLIFFS = Path(".cache/noncliff_pairs")
FIG_DIR = Path("figures/cliffs")

HEADLINE_DATASETS = (D3_DOPAMINE, THROMBIN, GSK3B)
NEURAL_SHORT_IDS = ("chemeleon", "mist_28M")

# Drop sweep targets that produce fewer than this many cliffs - bootstrap
# CIs at very small n are meaningless and the box would be a single point.
MIN_CLIFFS_FOR_SWEEP = 30
N_BOOTSTRAP = 1000


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


def _process_target(
    ds: MolACEDataset,
    chemeleon: CheMeleonFingerprint,
    mist_28m: MISTFingerprint,
    use_cache: bool,
    n_bootstrap: int,
    include_l2: bool,
) -> tuple[
    dict[str, CliffSeparationResult] | None,
    dict[str, CliffSeparationResult] | None,
]:
    """Run cliff finding + matched non-cliff sampling + per-FP separation
    for one target. Returns (cosine_results, l2_results), or (None, None)
    if the target produces no cliffs / non-cliffs.

    L2 results are only computed when include_l2 is True.
    """
    mols, y = _load_dataset(ds)
    cliff_pairs = _load_or_compute_cliffs(ds, mols, y, use_cache=use_cache)
    if not cliff_pairs:
        logger.warning(f"{ds.name}: no cliff pairs; skipping")
        return None, None

    noncliff_pairs = _load_or_sample_noncliffs(
        ds, mols, y, cliff_pairs, use_cache=use_cache,
    )
    if not noncliff_pairs:
        logger.warning(f"{ds.name}: no matched non-cliffs; skipping")
        return None, None

    fps = _build_fps(mols, chemeleon, mist_28m)
    cosine_sep = cliff_separation_for_all(
        fps, cliff_pairs, noncliff_pairs,
        n_bootstrap=n_bootstrap,
    )
    l2_sep: dict[str, CliffSeparationResult] | None = None
    if include_l2:
        l2_sep = cliff_separation_for_all(
            {k: v for k, v in fps.items() if k in NEURAL_SHORT_IDS},
            cliff_pairs,
            noncliff_pairs,
            extra_metrics={k: "euclidean" for k in NEURAL_SHORT_IDS},
            n_bootstrap=n_bootstrap,
        )
    del fps
    return cosine_sep, l2_sep


def main(use_cache: bool = True, skip_sweep: bool = False) -> None:
    device = _device()
    logger.info(f"using device={device}")

    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    # =============================================================
    # Layer 1: headline 3-target detail figures.
    # =============================================================
    cosine_results: dict[str, dict[str, CliffSeparationResult]] = {}
    l2_results: dict[str, dict[str, CliffSeparationResult]] = {}

    for ds in HEADLINE_DATASETS:
        logger.info(f"=== headline: {ds.name} ({ds.target_label}) ===")
        cos, l2 = _process_target(
            ds, chemeleon, mist_28m,
            use_cache=use_cache, n_bootstrap=N_BOOTSTRAP, include_l2=True,
        )
        if cos is not None:
            cosine_results[ds.target_label] = cos
        if l2 is not None:
            l2_results[ds.target_label] = l2

    if not cosine_results:
        logger.error("no headline targets produced results; exiting")
        return

    plot_pr_auc_summary(
        cosine_results,
        out_path=FIG_DIR / "04_pr_auc_summary.png",
        title=(
            "PR-AUC: cliff vs matched-non-cliff separation per fingerprint "
            "(\u00b1 95% bootstrap CI half-width)"
        ),
    )

    cliff_blind_by_dataset = {
        ds: {sid: r.cliff_blind_rate_07 for sid, r in res.items()}
        for ds, res in cosine_results.items()
    }
    pr_auc_by_dataset = {
        ds: {sid: r.pr_auc for sid, r in res.items()}
        for ds, res in cosine_results.items()
    }
    cliff_blind_ci_by_dataset = {
        ds: {sid: r.cliff_blind_rate_07_ci for sid, r in res.items()}
        for ds, res in cosine_results.items()
    }
    pr_auc_ci_by_dataset = {
        ds: {sid: r.pr_auc_ci for sid, r in res.items()}
        for ds, res in cosine_results.items()
    }
    plot_metric_scatter(
        cliff_blind_by_dataset,
        pr_auc_by_dataset,
        out_path=FIG_DIR / "05_metric_scatter.png",
        title=(
            "Absolute threshold (x) vs rank-based PR-AUC (y), "
            "with 95% bootstrap CIs"
        ),
        cliff_blind_ci_by_dataset=cliff_blind_ci_by_dataset,
        pr_auc_ci_by_dataset=pr_auc_ci_by_dataset,
    )

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

    if skip_sweep:
        logger.info("skipping cross-target sweep")
        return

    # =============================================================
    # Layer 2: cross-target sweep on all 30 MoleculeACE targets.
    # =============================================================
    logger.info("=== cross-target sweep ===")
    sweep_pr_auc: dict[str, dict[str, float]] = {}
    sweep_dropped: list[tuple[str, int]] = []

    for ds in ALL_MOLACE_DATASETS:
        if ds.target_label in cosine_results:
            # Reuse headline-target results - we already computed PR-AUC.
            r = cosine_results[ds.target_label]
            n_cliffs = next(iter(r.values())).n_cliffs
            if n_cliffs < MIN_CLIFFS_FOR_SWEEP:
                sweep_dropped.append((ds.target_label, n_cliffs))
                continue
            sweep_pr_auc[ds.target_label] = {sid: v.pr_auc for sid, v in r.items()}
            continue

        logger.info(f"=== sweep: {ds.name} ({ds.target_label}) ===")
        cos, _ = _process_target(
            ds, chemeleon, mist_28m,
            use_cache=use_cache, n_bootstrap=0, include_l2=False,
        )
        if cos is None:
            sweep_dropped.append((ds.target_label, 0))
            continue
        n_cliffs = next(iter(cos.values())).n_cliffs
        if n_cliffs < MIN_CLIFFS_FOR_SWEEP:
            sweep_dropped.append((ds.target_label, n_cliffs))
            continue
        sweep_pr_auc[ds.target_label] = {sid: v.pr_auc for sid, v in cos.items()}

    logger.info(
        f"sweep coverage: {len(sweep_pr_auc)} targets retained, "
        f"{len(sweep_dropped)} dropped"
    )
    for label, n in sweep_dropped:
        logger.info(f"  dropped: {label} (n_cliffs={n})")

    if sweep_pr_auc:
        plot_cross_target_boxplot(
            sweep_pr_auc,
            out_path=FIG_DIR / "07_pr_auc_cross_target_boxplot.png",
            title=(
                "PR-AUC across all MoleculeACE targets per fingerprint "
                f"(n={len(sweep_pr_auc)} targets, cliff n \u2265 {MIN_CLIFFS_FOR_SWEEP})"
            ),
            n_targets_label=f"{len(sweep_pr_auc)} targets, cliff n \u2265 {MIN_CLIFFS_FOR_SWEEP}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore cached cliff candidates and recompute",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="only do the 3-target headline figures, skip the cross-target sweep",
    )
    args = parser.parse_args()
    main(use_cache=not args.no_cache, skip_sweep=args.skip_sweep)
