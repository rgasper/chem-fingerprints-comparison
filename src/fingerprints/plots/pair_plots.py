"""Plotting utilities for the pair comparison figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D

from fingerprints.fingerprint_methods.base import FingerprintResult
from fingerprints.fingerprint_methods.similarity import (
    pairwise_similarity,
    similarity_label,
)


def _mol_to_image_array(mol: Mol, size: int = 200) -> np.ndarray:
    """Render an RDKit mol to an RGB numpy array."""
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.drawOptions().clearBackground = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("RGB")
    return np.asarray(img)


def plot_pair_similarity_grid(
    fps: dict[str, FingerprintResult],
    mols: list[Mol],
    labels: list[str],
    out_dir: Path,
    pair_boundaries: list[tuple[int, int]] | None = None,
    pair_descriptions: list[str] | None = None,
    panel_size: float = 16.0,
    structure_px: int = 480,
    dpi: int = 200,
) -> list[Path]:
    """One large similarity-heatmap figure per fingerprint method, with the
    diagonal cells filled with rendered molecule structures.

    Args:
        fps: dict from short fingerprint id to FingerprintResult
        mols: molecules in the same order as the fingerprint rows
        labels: human-readable labels (used for axis ticks)
        out_dir: directory to write `01_pair_similarity_<key>.png` into
        pair_boundaries: list of (lo, hi) tuples specifying intended pairs;
            each draws a 2x2 rectangle around (lo..hi, lo..hi)
        pair_descriptions: optional one-liner per pair (same length as
            pair_boundaries). When provided, rendered as a legend in the
            unused lower-triangle area.
        panel_size: matplotlib figure side in inches
        structure_px: pixel size of each rendered diagonal structure
        dpi: output DPI

    Returns:
        list of written file paths in the same order as fps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # render each molecule once
    structure_imgs = [_mol_to_image_array(m, size=structure_px) for m in mols]

    n = len(mols)
    for key, fp in fps.items():
        fig, ax = plt.subplots(figsize=(panel_size, panel_size))
        sim = pairwise_similarity(fp)
        # Pick colormap and value range based on fingerprint kind:
        # - binary -> Tanimoto in [0, 1], white -> blue (Blues, sequential)
        # - continuous -> cosine in [-1, 1], red -> white -> blue
        #   (coolwarm reversed so blue = high similarity, matching binary)
        if fp.kind == "continuous":
            vmin, vmax = -1.0, 1.0
            cmap = plt.get_cmap("coolwarm_r").copy()
        else:
            vmin, vmax = 0.0, 1.0
            cmap = plt.get_cmap("Blues").copy()

        # only display the upper triangle; mask diagonal and lower triangle
        # so the redundant lower-triangle cells don't add visual noise.
        sim_display = sim.copy().astype(float)
        mask = np.tril(np.ones_like(sim_display, dtype=bool), k=0)
        sim_display[mask] = np.nan
        cmap.set_bad(color="white")

        im = ax.imshow(sim_display, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{fp.name}  ({similarity_label(fp.kind)})", fontsize=18, pad=14
        )
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        # write similarity values in off-diagonal cells (upper triangle)
        for i in range(n):
            for j in range(n):
                if i < j:
                    val = sim[i, j]
                    # Pick text color so it's readable against the cell color.
                    # Both Blues and coolwarm_r are darkest at high values, so
                    # use white text only when the cell is dark.
                    if fp.kind == "continuous":
                        # darkest at val=+1 (deep blue) and val=-1 (deep red)
                        is_dark = abs(val) > 0.55
                    else:
                        # Blues sequential: darkest at val=1
                        is_dark = val > 0.6
                    color = "white" if is_dark else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color=color,
                    )

        # overlay structures on the diagonal cells
        for i in range(n):
            ax.imshow(
                structure_imgs[i],
                extent=(i - 0.5, i + 0.5, i + 0.5, i - 0.5),
                aspect="auto",
                zorder=2,
            )

        # outline the 2x2 block for each pair: encloses both diagonal
        # structure cells and the (lo, hi) similarity cell, making it visually
        # obvious which two molecules form the pair. Label the empty
        # lower-left corner of each block with A/B/C/D so the legend can
        # refer to the pairs unambiguously.
        if pair_boundaries is not None:
            for idx, (lo, hi) in enumerate(pair_boundaries):
                rect = plt.Rectangle(
                    (lo - 0.5, lo - 0.5),
                    hi - lo + 1,
                    hi - lo + 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.4,
                    zorder=3,
                )
                ax.add_patch(rect)
                ax.text(
                    lo,
                    hi,
                    chr(ord("A") + idx),
                    ha="center",
                    va="center",
                    fontsize=42,
                    fontweight="bold",
                    color="red",
                    zorder=4,
                )

        # Render pair descriptions as a legend in the unused lower-left
        # whitespace below all four 2x2 boxes. With pairs at rows 0/1, 2/3,
        # 4/5, 6/7 the boxes only consume columns <= row, so the lower-left
        # rectangle (rows 6-7 wide, cols 0-3) is reliably empty.
        if pair_descriptions is not None and pair_boundaries is not None:
            assert len(pair_descriptions) == len(pair_boundaries)
            n_pairs = len(pair_descriptions)
            line_height = 0.55
            # anchor near the very bottom-left corner of the matrix,
            # below the last pair's box
            block_top_row = n - 1.5 - (n_pairs - 1) * line_height
            for i, desc in enumerate(pair_descriptions):
                ax.text(
                    -0.35,
                    block_top_row + i * line_height,
                    desc,
                    ha="left",
                    va="center",
                    fontsize=11,
                    color="black",
                    zorder=4,
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        fig.tight_layout()
        out_path = out_dir / f"01_pair_similarity_{key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"wrote {out_path}")
        written.append(out_path)

    return written


def plot_pair_structures_strip(
    mols: list[Mol], labels: list[str], out_path: Path, mol_size: int = 220
) -> Path:
    """Render the molecules as a strip image (8 across) for use as a header / legend
    above the similarity grid.
    """
    n = len(mols)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4), squeeze=False)
    for i, (m, lab) in enumerate(zip(mols, labels)):
        img = _mol_to_image_array(m, size=mol_size)
        axes[0][i].imshow(img)
        axes[0][i].set_title(f"{i}: {lab}", fontsize=8)
        axes[0][i].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
