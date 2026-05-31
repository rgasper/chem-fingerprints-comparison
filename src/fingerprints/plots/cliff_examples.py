"""Plots for the worst-false-friend example figure.

Renders an 8-row table where each row shows one fingerprint's worst false
friend: a pair of molecules the fingerprint placed in each other's top-k
neighborhood despite being maximally activity-different. Each row carries
the structures (drawn with RDKit), the pKi values, the fingerprint's own
similarity score, and the neighbor rank (1 = nearest neighbor).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from loguru import logger
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D
from typeguard import typechecked

from fingerprints.clustering.cliffs import FalseFriendExample
from fingerprints.plots.groupings import grouped_order, style_for


def _draw_pair_image(
    mol_a: Mol,
    mol_b: Mol,
    size: tuple[int, int] = (700, 260),
) -> np.ndarray:
    """Render two molecules side by side into a single RGB array."""
    half_w = size[0] // 2
    h = size[1]

    def _draw_one(mol: Mol) -> Image.Image:
        drawer = rdMolDraw2D.MolDraw2DCairo(half_w, h)
        drawer.drawOptions().clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")

    img_a = _draw_one(mol_a)
    img_b = _draw_one(mol_b)
    composite = Image.new("RGB", size, "white")
    composite.paste(img_a, (0, 0))
    composite.paste(img_b, (half_w, 0))
    return np.asarray(composite)


@typechecked
def plot_worst_false_friend_examples(
    examples: dict[str, FalseFriendExample],
    mols: list[Mol],
    y: np.ndarray,
    out_path: Path,
    title: str = "",
    row_height: float = 1.7,
    dpi: int = 200,
) -> Path:
    """8-row figure: one row per fingerprint showing its worst false friend.

    Layout: rows ordered by GROUP_ORDER (Local atom-env / Path / Expert /
    Neural). Each row is a horizontal strip with:

        [fingerprint label, group-colored] | mol_i ... mol_j | metrics

    where 'metrics' lists the FP similarity, the neighbor rank (1..k), and
    |Delta y| in pKi units.

    Args:
        examples: mapping from fingerprint short id to its FalseFriendExample.
        mols: parallel list of RDKit Mol objects (indexed by example.i / .j).
        y: parallel pKi array.
        out_path: png destination.
        title: figure title.
        row_height: height in inches of each fingerprint row.
        dpi: output dpi.
    """
    short_ids = list(examples.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]
    n_rows = len(short_ids)

    fig_w = 12.0
    fig_h = row_height * n_rows + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Three columns per row: label (narrow), molecules (wide), metrics (narrow).
    gs = fig.add_gridspec(
        n_rows, 3, width_ratios=(1.6, 5.0, 1.4),
        hspace=0.15, wspace=0.05,
    )

    for ri, sid in enumerate(short_ids):
        ex = examples[sid]
        group, color, _ = style_for(sid)

        # Label cell
        ax_label = fig.add_subplot(gs[ri, 0])
        ax_label.axis("off")
        ax_label.text(
            0.05, 0.55, ex.name,
            transform=ax_label.transAxes,
            fontsize=11, fontweight="bold", color=color,
            ha="left", va="center",
        )
        ax_label.text(
            0.05, 0.25, group,
            transform=ax_label.transAxes,
            fontsize=8.5, color=color, ha="left", va="center",
        )

        # Structures cell
        ax_mol = fig.add_subplot(gs[ri, 1])
        img = _draw_pair_image(mols[ex.i], mols[ex.j])
        ax_mol.imshow(img)
        ax_mol.set_xticks([])
        ax_mol.set_yticks([])
        for spine in ax_mol.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.5)
        # pKi annotations under the two molecules
        ax_mol.text(
            0.25, -0.04, f"pKi = {y[ex.i]:.2f}",
            transform=ax_mol.transAxes,
            fontsize=9, ha="center", va="top",
        )
        ax_mol.text(
            0.75, -0.04, f"pKi = {y[ex.j]:.2f}",
            transform=ax_mol.transAxes,
            fontsize=9, ha="center", va="top",
        )

        # Metrics cell
        ax_metrics = fig.add_subplot(gs[ri, 2])
        ax_metrics.axis("off")
        fold_change = 10 ** ex.delta_y
        if fold_change >= 1000:
            fold_str = f"{fold_change/1000:.0f},000\u00d7"
        elif fold_change >= 100:
            fold_str = f"{fold_change:.0f}\u00d7"
        else:
            fold_str = f"{fold_change:.0f}\u00d7"
        # Use a monospace-ish layout to keep the columns aligned
        lines = [
            f"|\u0394pKi| = {ex.delta_y:.2f}",
            f"({fold_str} potency)",
            f"FP similarity = {ex.similarity:.2f}",
            f"neighbor rank = {ex.neighbor_rank}",
        ]
        for li, line in enumerate(lines):
            ax_metrics.text(
                0.0, 0.85 - li * 0.22, line,
                transform=ax_metrics.transAxes,
                fontsize=9, ha="left", va="center",
                fontweight="bold" if li == 0 else "normal",
                color="black" if li != 1 else "#555555",
            )

    if title:
        fig.suptitle(title, fontsize=13, y=0.995)

    # Don't call tight_layout - the imshow axes don't play well with it.
    # Use subplots_adjust for top margin if there is a title.
    fig.subplots_adjust(
        top=0.96 if title else 0.99,
        bottom=0.01,
        left=0.005, right=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
