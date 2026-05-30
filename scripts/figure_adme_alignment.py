"""Generate ADME alignment figures: kNN sweep curves + per-property bar charts.

Phase 1 (--mode sweep): runs the kNN sweep across k\u2208{1,3,5,10,20,50} for each
fingerprint on each ADME dataset, plus the Spearman \u03c1 between fingerprint
distance and property difference. Saves per-property line plots showing
how kNN performance varies with k. Use this to pick a single k that
plateaus across most fingerprints.

Phase 2 (--mode bars --k K): re-runs (or reuses cached) alignment metrics
and produces the final per-property bar chart at the chosen k, paired
with Spearman \u03c1.

Run:
  uv run python scripts/figure_adme_alignment.py --mode sweep
  uv run python scripts/figure_adme_alignment.py --mode bars --k 5
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.alignment import (
    AlignmentResult,
    alignment_for_all,
)
from fingerprints.data.tdc import (
    BBB_MARTINS,
    LIPOPHILICITY,
    SOLUBILITY,
    TDCDataset,
    load_tdc,
)
from fingerprints.fingerprint_methods import rdkit_fps
from fingerprints.fingerprint_methods.base import FingerprintResult
from fingerprints.fingerprint_methods.chemeleon_fp import CheMeleonFingerprint
from fingerprints.fingerprint_methods.mist_fp import MIST_28M, MISTFingerprint
from fingerprints.plots.alignment_plots import (
    plot_alignment_bars,
    plot_knn_sweep,
)


CACHE_TDC = Path(".cache/tdc")
CACHE_ALIGN = Path(".cache/alignment")
FIG_DIR = Path("figures/adme")

DEFAULT_KS = [1, 3, 5, 10, 20, 50]


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


def _load_dataset_with_fps(
    ds: TDCDataset,
    chemeleon: CheMeleonFingerprint,
    mist_28m: MISTFingerprint,
) -> tuple[dict[str, FingerprintResult], np.ndarray]:
    """Load TDC dataset, parse with RDKit, compute all 8 fingerprints."""
    df = load_tdc(ds, CACHE_TDC)
    smiles = df["smiles"].to_list()
    y = df["y"].to_numpy()
    pairs = [
        (m, yi) for m, yi in zip((MolFromSmiles(s) for s in smiles), y)
        if m is not None
    ]
    mols = [m for m, _ in pairs]
    y_clean = np.array([yi for _, yi in pairs], dtype=float)
    logger.info(f"{ds.name}: {len(mols)} mols parsed (out of {df.height})")
    fps = _build_fps(mols, chemeleon, mist_28m)
    return fps, y_clean


def _alignment_cache_path(ds_name: str) -> Path:
    return CACHE_ALIGN / f"{ds_name}.pkl"


def _compute_or_load_alignments(
    ds: TDCDataset,
    chemeleon: CheMeleonFingerprint,
    mist_28m: MISTFingerprint,
    ks: list[int],
    use_cache: bool = True,
) -> dict[str, AlignmentResult]:
    cache_path = _alignment_cache_path(ds.name)
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            cached: dict[str, AlignmentResult] = pickle.load(fh)
        # Validate that the cached ks contain everything we asked for
        cached_ks = next(iter(cached.values())).ks
        if all(k in cached_ks for k in ks):
            logger.info(f"reusing cached alignments at {cache_path}")
            return cached
        logger.info(
            f"cache at {cache_path} has ks={cached_ks}, requested {ks}; "
            "recomputing"
        )

    task_type = "regression" if ds.task_type == "regression" else "classification"
    fps, y = _load_dataset_with_fps(ds, chemeleon, mist_28m)
    aligns = alignment_for_all(fps, y, task_type=task_type, ks=ks)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(aligns, fh)
    logger.info(f"cached alignments to {cache_path}")
    return aligns


def main_sweep() -> None:
    device = _device()
    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    for ds in (SOLUBILITY, LIPOPHILICITY, BBB_MARTINS):
        aligns = _compute_or_load_alignments(
            ds, chemeleon, mist_28m, ks=DEFAULT_KS, use_cache=True,
        )
        score_label = "R\u00b2" if ds.task_type == "regression" else "ROC-AUC"
        plot_knn_sweep(
            aligns,
            out_path=FIG_DIR / f"02_knn_sweep_{ds.name}.png",
            title=f"kNN {score_label} vs k for {ds.property_label}",
        )


def main_bars(k: int) -> None:
    device = _device()
    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    ks = sorted(set(DEFAULT_KS + [k]))

    for ds in (SOLUBILITY, LIPOPHILICITY, BBB_MARTINS):
        aligns = _compute_or_load_alignments(
            ds, chemeleon, mist_28m, ks=ks, use_cache=True,
        )
        plot_alignment_bars(
            aligns,
            k_chosen=k,
            out_path=FIG_DIR / f"03_alignment_{ds.name}.png",
            title=(
                f"ADME alignment per fingerprint, {ds.property_label} "
                f"(k={k}, n_pairs=50k)"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("sweep", "bars"),
        default="sweep",
        help="sweep = kNN curves; bars = final bar charts at chosen k",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="k value for the bars phase"
    )
    args = parser.parse_args()
    if args.mode == "sweep":
        main_sweep()
    else:
        main_bars(args.k)
