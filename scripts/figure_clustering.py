"""Generate the clustering figure: 8-panel UMAP grid on a ChEMBL drug-like sample.

Each panel is one fingerprint's UMAP, colored by HDBSCAN cluster id.

Run:
  uv run python scripts/figure_clustering.py
  uv run python scripts/figure_clustering.py --n 5000  # smaller sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.embed import embed_all
from fingerprints.data.chembl import (
    download_chemreps,
    parse_chemreps,
    sample_valid_smiles,
)
from fingerprints.fingerprint_methods import rdkit_fps
from fingerprints.fingerprint_methods.base import FingerprintResult
from fingerprints.fingerprint_methods.chemeleon_fp import CheMeleonFingerprint
from fingerprints.fingerprint_methods.mist_fp import MIST_28M, MISTFingerprint
from fingerprints.plots.umap_grid import plot_umap_grid


CACHE_DIR = Path(".cache")
FIG_DIR = Path("figures/clustering")


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


def main(n: int = 10000, seed: int = 0) -> None:
    device = _device()
    logger.info(f"using device={device}")

    # 1. Get ChEMBL sample
    chemreps_path = CACHE_DIR / "chembl" / "chembl_chemreps.txt.gz"
    download_chemreps(chemreps_path)
    df_all = parse_chemreps(chemreps_path)
    logger.info(f"parsed {df_all.height} ChEMBL entries")
    sample = sample_valid_smiles(df_all, n=n, seed=seed)
    logger.info(f"sampled {sample.height} drug-like molecules")
    smiles = sample["canonical_smiles"].to_list()
    mols = [MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]
    logger.info(f"parsed {len(mols)} molecules with RDKit")

    # 2. Compute all 8 fingerprints
    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)
    fps = _build_fps(mols, chemeleon, mist_28m)

    # 3. UMAP + HDBSCAN per fingerprint
    embeddings = embed_all(
        fps,
        do_clustering=True,
        n_neighbors=30,
        min_dist=0.1,
        min_cluster_size=max(20, len(mols) // 200),  # ~50 for 10k sample
    )

    # 4. Plot
    plot_umap_grid(
        embeddings,
        out_path=FIG_DIR / f"01_umap_clusters_n{len(mols)}.png",
        color_kind="cluster",
        title=(
            f"UMAP of {len(mols)} ChEMBL drug-like molecules, "
            "per fingerprint (HDBSCAN clusters)"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="number of molecules")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n=args.n, seed=args.seed)
