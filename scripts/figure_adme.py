"""Generate ADME UMAP figures: 8-panel UMAPs colored by ADME property.

Each figure shows the same 8 fingerprints, each fit independently with UMAP
on the dataset's molecules, with points colored by the ADME measurement.

Three datasets:
  - Solubility (AqSolDB) - regression, log S
  - Lipophilicity (AstraZeneca) - regression, log D
  - BBB (Martins) - binary classification

Run:
  uv run python scripts/figure_adme.py            # all three datasets
  uv run python scripts/figure_adme.py --dataset bbb_martins
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.embed import embed_all
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
from fingerprints.plots.umap_grid import plot_umap_grid


CACHE_DIR = Path(".cache/tdc")
FIG_DIR = Path("figures/adme")


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


def run_one(
    ds: TDCDataset,
    chemeleon: CheMeleonFingerprint,
    mist_28m: MISTFingerprint,
    max_n: int | None = None,
) -> Path:
    logger.info(f"=== ADME UMAP: {ds.name} ===")
    df = load_tdc(ds, CACHE_DIR)
    if max_n is not None and df.height > max_n:
        df = df.sample(n=max_n, seed=0)
        logger.info(f"subsampled to {df.height} rows")

    smiles = df["smiles"].to_list()
    y = df["y"].to_numpy()

    # Parse with RDKit; drop molecules that fail to parse
    mols_with_y = [
        (m, yi) for m, yi in zip(
            (MolFromSmiles(s) for s in smiles), y
        ) if m is not None
    ]
    mols = [m for m, _ in mols_with_y]
    import numpy as np
    y_clean = np.array([yi for _, yi in mols_with_y], dtype=float)
    logger.info(f"{len(mols)} molecules parsed (out of {df.height})")

    fps = _build_fps(mols, chemeleon, mist_28m)

    # No HDBSCAN needed - we're coloring by property, not by cluster
    embeddings = embed_all(
        fps,
        do_clustering=False,
        n_neighbors=30,
        min_dist=0.1,
    )

    if ds.task_type == "regression":
        # AqSolDB has a long tail of very-low-solubility molecules; show
        # the bulk distribution by clipping color range to robust quantiles
        finite = y_clean[np.isfinite(y_clean)]
        vlo = float(np.quantile(finite, 0.01))
        vhi = float(np.quantile(finite, 0.99))
        # clip the color values to the robust range so the colormap is
        # not dominated by outliers
        y_for_color = np.clip(y_clean, vlo, vhi)
        out_path = plot_umap_grid(
            embeddings,
            out_path=FIG_DIR / f"01_umap_{ds.name}.png",
            color_values=y_for_color,
            color_kind="continuous",
            color_label=ds.property_label,
            cmap="viridis",
            title=(
                f"UMAP per fingerprint, {len(mols)} molecules, "
                f"colored by {ds.property_label}"
            ),
        )
    else:  # classification
        out_path = plot_umap_grid(
            embeddings,
            out_path=FIG_DIR / f"01_umap_{ds.name}.png",
            color_values=y_clean.astype(int),
            color_kind="binary",
            color_label=ds.property_label,
            title=(
                f"UMAP per fingerprint, {len(mols)} molecules, "
                f"colored by {ds.property_label}"
            ),
        )
    return out_path


def main(dataset: str | None = None, max_n: int | None = None) -> None:
    device = _device()
    logger.info(f"using device={device}")

    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    name_to_ds = {
        SOLUBILITY.name: SOLUBILITY,
        LIPOPHILICITY.name: LIPOPHILICITY,
        BBB_MARTINS.name: BBB_MARTINS,
    }
    if dataset is None:
        targets = [SOLUBILITY, LIPOPHILICITY, BBB_MARTINS]
    else:
        if dataset not in name_to_ds:
            raise SystemExit(
                f"unknown dataset {dataset!r}; choose one of {list(name_to_ds)}"
            )
        targets = [name_to_ds[dataset]]

    for ds in targets:
        run_one(ds, chemeleon, mist_28m, max_n=max_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        help="run a single dataset (e.g. solubility_aqsoldb); default = all",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="optional max molecules per dataset (for quick iteration)",
    )
    args = parser.parse_args()
    main(dataset=args.dataset, max_n=args.max_n)
