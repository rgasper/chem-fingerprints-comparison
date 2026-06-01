"""Generate activity-cliff figures on three MoleculeACE targets.

Activity cliffs are defined fingerprint-agnostically: pairs of molecules
with |delta pKi| >= 2.0 and graph_distance <= 5 (where graph_distance =
n_atoms_i + n_atoms_j - 2 * mcs_atoms). For each fingerprint we measure
its similarity over the cliff pair set and produce:

- 01_cliff_similarity_violins_<target>.png: distribution of similarities
  per fingerprint per dataset. Lower = better.
- 02_cliff_examples_<short_id>.png: per-FP deep-dive showing top-3 most
  cliff-blind and top-3 most cliff-aware molecule pairs across all
  three datasets. One image file per fingerprint.
- 03_cliff_blind_summary.png: cross-dataset heatmap of P(sim >= 0.7).

Cliff candidates are cached at .cache/cliff_pairs/<dataset>.pkl since the
MCS verification step is expensive (~10s for GSK-3 beta, ~1 min for
Thrombin/D3 with thread-parallel MCS).

Run:
  uv run python scripts/figure_cliffs.py
  uv run python scripts/figure_cliffs.py --no-cache  # recompute candidates
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.cliffs import (
    CliffExamplePair,
    CliffPair,
    CliffSimResult,
    cliff_similarity_for_all,
    find_cliff_pairs,
    select_cliff_examples_for_all,
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
from fingerprints.plots.cliffs import (
    plot_cliff_blind_summary,
    plot_cliff_examples_per_fp,
    plot_cliff_similarity_violins,
)


CACHE_MOLACE = Path(".cache/molace")
CACHE_CLIFFS = Path(".cache/cliff_pairs")
FIG_DIR = Path("figures/cliffs")

DATASETS = (D3_DOPAMINE, THROMBIN, GSK3B)


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


def main(use_cache: bool = True) -> None:
    device = _device()
    logger.info(f"using device={device}")

    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    # Collect per-dataset results, plus per-fingerprint examples spanning
    # all datasets so we can emit one figure per FP at the end.
    sims_by_dataset: dict[str, dict[str, CliffSimResult]] = {}
    mols_by_dataset: dict[str, list] = {}
    y_by_dataset: dict[str, np.ndarray] = {}
    # examples_by_fp[short_id][dataset_label] = (most_blind, most_aware) lists
    examples_by_fp: dict[
        str, dict[str, tuple[list[CliffExamplePair], list[CliffExamplePair]]]
    ] = {}
    fp_display_names: dict[str, str] = {}

    for ds in DATASETS:
        logger.info(f"=== {ds.name} ({ds.target_label}) ===")
        mols, y = _load_dataset(ds)
        cliff_pairs = _load_or_compute_cliffs(ds, mols, y, use_cache=use_cache)

        if not cliff_pairs:
            logger.warning(f"{ds.name}: no cliff pairs found, skipping")
            continue

        fps = _build_fps(mols, chemeleon, mist_28m)
        sims = cliff_similarity_for_all(fps, cliff_pairs)
        sims_by_dataset[ds.target_label] = sims
        mols_by_dataset[ds.target_label] = mols
        y_by_dataset[ds.target_label] = y

        examples = select_cliff_examples_for_all(fps, cliff_pairs, n_top=3)
        for sid, fp in fps.items():
            examples_by_fp.setdefault(sid, {})[ds.target_label] = examples[sid]
            fp_display_names[sid] = fp.name

        plot_cliff_similarity_violins(
            sims,
            out_path=FIG_DIR / f"01_cliff_similarity_violins_{ds.name}.png",
            title=(
                f"Cliff-pair similarity distribution \u2014 {ds.target_label} "
                f"({ds.target_class}, n={len(cliff_pairs)} cliff pairs)"
            ),
        )

        del fps

    # Per-fingerprint deep-dive: one image per FP, six panels per dataset
    # (3 most cliff-blind + 3 most cliff-aware).
    for sid, examples_by_dataset in examples_by_fp.items():
        plot_cliff_examples_per_fp(
            short_id=sid,
            fp_display_name=fp_display_names[sid],
            examples_by_dataset=examples_by_dataset,
            mols_by_dataset=mols_by_dataset,
            y_by_dataset=y_by_dataset,
            out_path=FIG_DIR / f"02_cliff_examples_{sid}.png",
            title=(
                f"Cliff examples \u2014 {fp_display_names[sid]}"
            ),
        )

    # Cross-dataset summary
    plot_cliff_blind_summary(
        sims_by_dataset,
        threshold=0.7,
        out_path=FIG_DIR / "03_cliff_blind_summary.png",
        title=(
            "Cliff-blind rate across datasets "
            "(graph-distance \u2264 5, |\u0394pKi| \u2265 2.0)"
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
