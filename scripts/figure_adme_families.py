"""ADME UMAPs zoomed in on 1-2 chemical families per dataset.

Same UMAPs as figure_adme.py, but the background is greyed out and only
points belonging to the selected scaffold families are colored by the
property value. This shows where each fingerprint lays out members of one
chemical family relative to each other and to the rest of the dataset.

Families are picked automatically: the Bemis-Murcko scaffolds with the
largest size * property-range product (subject to a minimum size). For
binary tasks, range collapses to 1 if both classes present, so this picks
the biggest scaffolds with class diversity.

Run:
  uv run python scripts/figure_adme_families.py
  uv run python scripts/figure_adme_families.py --dataset solubility_aqsoldb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.embed import embed_all
from fingerprints.clustering.scaffolds import scaffold_ids, select_top_families
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
from fingerprints.plots.umap_grid import plot_umap_grid_family_overlay


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


def _short_family_label(scaffold_smiles: str, max_len: int = 40) -> str:
    """Compact label for the legend; SMILES truncated with ellipsis."""
    if len(scaffold_smiles) <= max_len:
        return scaffold_smiles
    return scaffold_smiles[: max_len - 1] + "..."


def run_one(
    ds: TDCDataset,
    chemeleon: CheMeleonFingerprint,
    mist_28m: MISTFingerprint,
    n_families: int,
    min_family_size: int,
    min_scaffold_atoms: int,
    max_n: int | None = None,
) -> Path | None:
    logger.info(f"=== ADME family overlay: {ds.name} ===")
    df = load_tdc(ds, CACHE_DIR)
    if max_n is not None and df.height > max_n:
        df = df.sample(n=max_n, seed=0)
        logger.info(f"subsampled to {df.height} rows")

    smiles = df["smiles"].to_list()
    y = df["y"].to_numpy()

    mols_with_y = [
        (m, yi)
        for m, yi in zip((MolFromSmiles(s) for s in smiles), y)
        if m is not None
    ]
    mols = [m for m, _ in mols_with_y]
    y_clean = np.array([yi for _, yi in mols_with_y], dtype=float)
    logger.info(f"{len(mols)} molecules parsed (out of {df.height})")

    # scaffolds + family selection
    scaffolds = scaffold_ids(mols)
    chosen = select_top_families(
        scaffolds, y_clean,
        n_families=n_families,
        min_size=min_family_size,
        min_scaffold_atoms=min_scaffold_atoms,
    )
    if not chosen:
        logger.warning(
            f"no scaffold meets min_size={min_family_size} on {ds.name}; "
            "skipping family-overlay figure"
        )
        return None

    # family membership per molecule (-1 = background)
    scaf_arr = np.array(scaffolds)
    family_membership = np.full(len(mols), -1, dtype=int)
    for fi, scaf in enumerate(chosen):
        family_membership[scaf_arr == scaf] = fi

    family_labels = [_short_family_label(s) for s in chosen]
    for fi, lbl in enumerate(family_labels):
        n_in = int((family_membership == fi).sum())
        logger.info(f"family {fi}: n={n_in} {lbl}")

    fps = _build_fps(mols, chemeleon, mist_28m)
    embeddings = embed_all(
        fps,
        do_clustering=False,
        n_neighbors=30,
        min_dist=0.1,
    )

    if ds.task_type == "regression":
        # Use full (unclipped) range here since we're only plotting family
        # members - they tend to span a narrower range than the dataset
        # outliers we clipped against in figure_adme.py.
        out_path = plot_umap_grid_family_overlay(
            embeddings,
            out_path=FIG_DIR / f"04_umap_families_{ds.name}.png",
            family_membership=family_membership,
            color_values=y_clean,
            color_kind="continuous",
            family_labels=family_labels,
            color_label=ds.property_label,
            cmap="viridis",
            title=(
                f"UMAP per fingerprint, {len(mols)} molecules; "
                f"top {len(chosen)} scaffold families colored by "
                f"{ds.property_label}"
            ),
        )
    else:
        out_path = plot_umap_grid_family_overlay(
            embeddings,
            out_path=FIG_DIR / f"04_umap_families_{ds.name}.png",
            family_membership=family_membership,
            color_values=y_clean.astype(int),
            color_kind="binary",
            family_labels=family_labels,
            color_label=ds.property_label,
            title=(
                f"UMAP per fingerprint, {len(mols)} molecules; "
                f"top {len(chosen)} scaffold families colored by "
                f"{ds.property_label}"
            ),
        )
    return out_path


def main(
    dataset: str | None = None,
    max_n: int | None = None,
    n_families: int = 2,
    min_family_size: int = 15,
    min_scaffold_atoms: int = 12,
) -> None:
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
        run_one(
            ds, chemeleon, mist_28m,
            n_families=n_families,
            min_family_size=min_family_size,
            min_scaffold_atoms=min_scaffold_atoms,
            max_n=max_n,
        )


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
    parser.add_argument(
        "--n-families",
        type=int,
        default=2,
        help="number of scaffold families to overlay (1-4)",
    )
    parser.add_argument(
        "--min-family-size",
        type=int,
        default=15,
        help="minimum scaffold members for a family to be considered",
    )
    parser.add_argument(
        "--min-scaffold-atoms",
        type=int,
        default=12,
        help=(
            "minimum heavy atoms in the scaffold itself; filters out "
            "generic single-ring scaffolds like benzene. set to 0 to disable."
        ),
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        max_n=args.max_n,
        n_families=args.n_families,
        min_family_size=args.min_family_size,
        min_scaffold_atoms=args.min_scaffold_atoms,
    )
