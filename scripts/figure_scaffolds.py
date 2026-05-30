"""Generate scaffold-hopping figures.

Two figures:

1. `01_scaffold_purity_n{n}.png` \u2014 bar chart of scaffold purity at k=5 for
   each fingerprint, on a ChEMBL drug-like sample. A dashed line shows the
   random-match baseline (= same-scaffold rate in the sample). Bars far
   above baseline indicate the fingerprint preferentially groups molecules
   by scaffold.

2. `02_purity_vs_property_pareto.png` \u2014 scatter of (scaffold purity, property
   kNN R\u00b2/ROC-AUC) at k=5, one panel per ADME property. Reads as a Pareto:
   upper-left (high property alignment, low scaffold dependence) is the
   useful corner. Reuses the cached alignment results from
   `figure_adme_alignment.py`.

Run:
  uv run python scripts/figure_scaffolds.py            # both figures, n=10000
  uv run python scripts/figure_scaffolds.py --n 5000   # smaller / faster
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
from fingerprints.clustering.scaffolds import (
    scaffold_ids,
    scaffold_purity_for_all,
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
from fingerprints.plots.scaffolds import (
    plot_purity_property_pareto,
    plot_scaffold_purity_bars,
)


CACHE_DIR = Path(".cache")
CACHE_TDC = Path(".cache/tdc")
CACHE_ALIGN = Path(".cache/alignment")
FIG_DIR = Path("figures/scaffolds")

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


def _load_dataset_mols_with_y(ds: TDCDataset) -> tuple[list, np.ndarray]:
    df = load_tdc(ds, CACHE_TDC)
    smiles = df["smiles"].to_list()
    y = df["y"].to_numpy()
    pairs = [
        (m, yi) for m, yi in zip((MolFromSmiles(s) for s in smiles), y)
        if m is not None
    ]
    mols = [m for m, _ in pairs]
    y_clean = np.array([yi for _, yi in pairs], dtype=float)
    return mols, y_clean


def _load_alignments_cached(ds: TDCDataset) -> dict[str, AlignmentResult] | None:
    cache_path = CACHE_ALIGN / f"{ds.name}.pkl"
    if not cache_path.exists():
        logger.warning(
            f"no cached alignment at {cache_path}; will recompute"
        )
        return None
    with open(cache_path, "rb") as fh:
        return pickle.load(fh)


def main_bars(k: int) -> None:
    """Scaffold purity bar chart on AqSolDB.

    AqSolDB has dense scaffold repetition (~86% of molecules sit in non-
    singleton scaffold groups), which makes the metric discriminative.
    A random ChEMBL sample by contrast is too scaffold-diverse for this
    figure (almost all scaffolds are singletons; purity hovers near
    baseline regardless of fingerprint).
    """
    device = _device()
    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    ds = SOLUBILITY
    mols, _ = _load_dataset_mols_with_y(ds)
    fps = _build_fps(mols, chemeleon, mist_28m)

    scaffolds = scaffold_ids(mols)
    purities = scaffold_purity_for_all(fps, scaffolds, ks=DEFAULT_KS)
    n_unique = next(iter(purities.values())).n_unique_scaffolds
    baseline = next(iter(purities.values())).baseline
    logger.info(
        f"scaffold sample: n={len(mols)}, "
        f"n_unique_scaffolds={n_unique}, baseline={baseline:.4f}"
    )

    plot_scaffold_purity_bars(
        purities,
        k=k,
        out_path=FIG_DIR / f"01_scaffold_purity_k{k}_{ds.name}.png",
        title=(
            f"Scaffold purity at k={k}, {ds.property_label} "
            f"({len(mols)} molecules, {n_unique} unique scaffolds)"
        ),
    )


def main_pareto(k: int) -> None:
    """Pareto: scaffold purity vs property kNN, per ADME property.

    For each property: load the dataset, compute fingerprints + scaffold ids
    on the dataset's molecules, and pull cached alignments. Both metrics use
    the same k.
    """
    device = _device()
    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    alignments_by_property: dict[str, dict[str, AlignmentResult]] = {}
    purities_by_property: dict[str, dict] = {}

    for ds in (SOLUBILITY, LIPOPHILICITY, BBB_MARTINS):
        mols, y = _load_dataset_mols_with_y(ds)

        # Compute fingerprints once per dataset and reuse for both alignment
        # (if not cached) and scaffold purity. Avoids re-running MIST twice.
        fps = _build_fps(mols, chemeleon, mist_28m)

        # Alignment: prefer cache, otherwise compute (also caches it)
        cache_path = CACHE_ALIGN / f"{ds.name}.pkl"
        aligns = _load_alignments_cached(ds)
        if aligns is None or k not in next(iter(aligns.values())).ks:
            logger.info(f"computing alignment for {ds.name}")
            task_type = (
                "regression" if ds.task_type == "regression" else "classification"
            )
            aligns = alignment_for_all(
                fps, y, task_type=task_type, ks=DEFAULT_KS,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(aligns, fh)

        # Scaffold purity on the same molecules / fingerprints
        scaffolds = scaffold_ids(mols)
        purities = scaffold_purity_for_all(fps, scaffolds, ks=DEFAULT_KS)

        alignments_by_property[ds.property_label] = aligns
        purities_by_property[ds.property_label] = purities

        # Free the fingerprint arrays before the next dataset to keep peak
        # memory bounded on small machines.
        del fps

    plot_purity_property_pareto(
        purities_by_property=purities_by_property,
        alignments_by_property=alignments_by_property,
        k=k,
        out_path=FIG_DIR / f"02_purity_vs_property_pareto_k{k}.png",
    )


def main(k: int, mode: str) -> None:
    if mode in ("bars", "both"):
        main_bars(k=k)
    if mode in ("pareto", "both"):
        main_pareto(k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("bars", "pareto", "both"), default="both",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(k=args.k, mode=args.mode)
