"""Generate the pair-comparison figures for both small and drug-sized molecule sets.

Run:
  uv run python scripts/figure_pairs.py            # both sets, MIST 28M only
  uv run python scripts/figure_pairs.py --with-1.8B  # also include MIST 1.8B (large download)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from rdkit.Chem import Mol, MolFromSmiles

from fingerprints.data.pairs import PAIRS, all_labels, all_smiles
from fingerprints.data.pairs_drug import PAIRS_DRUG, all_labels_drug, all_smiles_drug
from fingerprints.fingerprint_methods import rdkit_fps
from fingerprints.fingerprint_methods.base import FingerprintResult
from fingerprints.fingerprint_methods.chemeleon_fp import CheMeleonFingerprint
from fingerprints.fingerprint_methods.mist_fp import MIST_28M, MIST_1_8B, MISTFingerprint
from fingerprints.plots.pair_plots import plot_pair_similarity_grid
from fingerprints.plots.pair_summary import plot_pair_summary_bars


FIG_DIR = Path("figures/pairs")


@dataclass(frozen=True)
class PairSet:
    name: str
    pairs: tuple
    smiles: list[str]
    labels: list[str]


SMALL = PairSet("small", PAIRS, all_smiles(), all_labels())
DRUG = PairSet("drug", PAIRS_DRUG, all_smiles_drug(), all_labels_drug())


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _build_fps(
    mols: list[Mol],
    chemeleon: CheMeleonFingerprint,
    mist_28m: MISTFingerprint,
    mist_1_8b: MISTFingerprint | None,
) -> dict[str, FingerprintResult]:
    fps: dict[str, FingerprintResult] = dict(rdkit_fps.all_classical(mols))
    fps["chemeleon"] = chemeleon(mols)
    fps["mist_28M"] = mist_28m(mols)
    if mist_1_8b is not None:
        fps["mist_1_8B"] = mist_1_8b(mols)
    return fps


def main(include_mist_1_8b: bool = False) -> None:
    device = _device()
    logger.info(f"using device={device}")

    # Load neural models once and reuse across both pair sets
    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)
    mist_1_8b: MISTFingerprint | None = None
    if include_mist_1_8b:
        logger.info("loading MIST-1.8B (large download / slow inference)")
        mist_1_8b = MISTFingerprint(
            model_id=MIST_1_8B, device=device, batch_size=4
        )

    for ps in (SMALL, DRUG):
        logger.info(f"=== pair set: {ps.name} ===")
        mols = [MolFromSmiles(s) for s in ps.smiles]
        out_dir = FIG_DIR / ps.name

        fps = _build_fps(mols, chemeleon, mist_28m, mist_1_8b)
        pair_boundaries = [(2 * i, 2 * i + 1) for i in range(len(ps.pairs))]
        # General categorical descriptions (the same for both sets) so the
        # audience can read the figure as "this fingerprint thinks pair X is
        # this similar" not "this fingerprint thinks 4-Br vs 4-Cl is..."
        category_descriptions = [
            "Pair A: same scaffold, minor decoration change",
            "Pair B: same scaffold, major decoration change",
            "Pair C: different scaffold, similar decorations",
            "Pair D: different scaffold and decorations",
        ]
        plot_pair_similarity_grid(
            fps,
            mols,
            ps.labels,
            out_dir,
            pair_boundaries=pair_boundaries,
            pair_descriptions=category_descriptions,
        )

        # Summary bar chart of within-pair distance per fingerprint
        bar_labels = [
            "A\nsame scaffold,\nminor change",
            "B\nsame scaffold,\nmajor change",
            "C\ndifferent scaffold,\nsimilar decorations",
            "D\ndifferent scaffold\nand decorations",
        ]
        plot_pair_summary_bars(
            fps,
            pair_boundaries=pair_boundaries,
            pair_labels=bar_labels,
            out_path=out_dir / "03_within_pair_distance_summary.png",
            mols=mols,
            mol_labels=ps.labels,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-1.8B", action="store_true", dest="with_big")
    args = parser.parse_args()
    main(include_mist_1_8b=args.with_big)
