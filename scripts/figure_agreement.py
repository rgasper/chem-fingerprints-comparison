"""Generate the fingerprint agreement heatmap (RV coefficients).

Computes pairwise RV between all 8 fingerprints on a ChEMBL drug-like sample
and writes a single k x k heatmap. Reuses the same molecule sample as the
clustering figure for cross-figure consistency.

Run:
  uv run python scripts/figure_agreement.py
  uv run python scripts/figure_agreement.py --n 2000  # smaller / faster
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.agreement import rv_matrix
from fingerprints.data.chembl import (
    download_chemreps,
    parse_chemreps,
    sample_valid_smiles,
)
from fingerprints.fingerprint_methods import rdkit_fps
from fingerprints.fingerprint_methods.base import FingerprintResult
from fingerprints.fingerprint_methods.chemeleon_fp import CheMeleonFingerprint
from fingerprints.fingerprint_methods.mist_fp import MIST_28M, MISTFingerprint
from fingerprints.plots.agreement import plot_rv_heatmap


CACHE_DIR = Path(".cache")
FIG_DIR = Path("figures/agreement")


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


def main(n: int = 5000, seed: int = 0) -> None:
    device = _device()
    logger.info(f"using device={device}")

    chemreps_path = CACHE_DIR / "chembl" / "chembl_chemreps.txt.gz"
    download_chemreps(chemreps_path)
    df_all = parse_chemreps(chemreps_path)
    sample = sample_valid_smiles(df_all, n=n, seed=seed)
    smiles = sample["canonical_smiles"].to_list()
    mols = [MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]
    logger.info(f"using {len(mols)} molecules")

    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)
    fps = _build_fps(mols, chemeleon, mist_28m)

    result = rv_matrix(fps)

    out_path = FIG_DIR / f"01_rv_agreement_n{len(mols)}.png"
    plot_rv_heatmap(
        result,
        out_path=out_path,
        title=(
            f"Fingerprint agreement (RV coefficient), {len(mols)} ChEMBL molecules"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n=args.n, seed=args.seed)
