"""Plots for the worst-false-friend and best-catch example figures.

Renders a per-fingerprint table of two molecule-pair illustrations: on the
left the worst false friend (a top-k neighbor pair with the largest
|Delta y| - the FP's biggest 'I thought these were similar but they
aren't' miss); on the right the best catch (an MCS-verified cliff pair
the FP correctly placed outside both molecules' top-k - its biggest
'I correctly didn't friend them' win).

Pairs come from analysis functions in `clustering.cliffs`.
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

from fingerprints.clustering.cliffs import (
    BestCatchExample,
    FalseFriendExample,
)
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


def _format_fold(delta_y: float) -> str:
    fold = 10 ** delta_y
    if fold >= 1000:
        return f"{fold/1000:.0f},000\u00d7"
    return f"{fold:.0f}\u00d7"


def _draw_pair_cell(
    ax_mol,
    ax_metrics,
    mol_a: Mol,
    mol_b: Mol,
    pki_a: float,
    pki_b: float,
    metric_lines: list[tuple[str, bool]],
) -> None:
    """Populate a (molecules, metrics) cell pair.

    Each metric_lines entry is (text, bold).
    """
    img = _draw_pair_image(mol_a, mol_b)
    ax_mol.imshow(img)
    ax_mol.set_xticks([])
    ax_mol.set_yticks([])
    for spine in ax_mol.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.5)
    ax_mol.text(
        0.25, -0.04, f"pKi = {pki_a:.2f}",
        transform=ax_mol.transAxes,
        fontsize=9, ha="center", va="top",
    )
    ax_mol.text(
        0.75, -0.04, f"pKi = {pki_b:.2f}",
        transform=ax_mol.transAxes,
        fontsize=9, ha="center", va="top",
    )

    ax_metrics.axis("off")
    n_lines = len(metric_lines)
    for li, (line, bold) in enumerate(metric_lines):
        # Distribute lines vertically with consistent spacing
        y_pos = 0.85 - li * (0.7 / max(n_lines - 1, 1))
        ax_metrics.text(
            0.0, y_pos, line,
            transform=ax_metrics.transAxes,
            fontsize=8.5, ha="left", va="center",
            fontweight="bold" if bold else "normal",
            color="black" if bold else "#444444",
        )


def _draw_no_catch_cell(ax_mol, ax_metrics) -> None:
    """Render a 'no catch' placeholder when an FP couldn't catch any cliff."""
    ax_mol.set_xticks([])
    ax_mol.set_yticks([])
    for spine in ax_mol.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.5)
    ax_mol.set_facecolor("#f5f5f5")
    ax_mol.text(
        0.5, 0.5, "no catch",
        transform=ax_mol.transAxes,
        fontsize=11, ha="center", va="center",
        color="#888888", style="italic",
    )
    ax_metrics.axis("off")
    ax_metrics.text(
        0.0, 0.5, "every cliff in\ncandidate set\nfell into top-k",
        transform=ax_metrics.transAxes,
        fontsize=8.5, ha="left", va="center",
        color="#888888", style="italic",
    )


@typechecked
def plot_cliff_examples(
    false_friends: dict[str, FalseFriendExample],
    best_catches: dict[str, BestCatchExample | None],
    mols: list[Mol],
    y: np.ndarray,
    out_path: Path,
    title: str = "",
    row_height: float = 1.7,
    dpi: int = 200,
) -> Path:
    """8-row x 2-column figure: worst false friend on left, best catch on right.

    Both columns share the same fingerprint ordering (by GROUP_ORDER).
    """
    short_ids = list(false_friends.keys())
    if list(best_catches.keys()) != short_ids:
        raise ValueError("false_friends and best_catches must have the same keys")
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]
    n_rows = len(short_ids)

    fig_w = 18.0
    fig_h = row_height * n_rows + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Per-row layout: label | ff-mols | ff-metrics | catch-mols | catch-metrics
    gs = fig.add_gridspec(
        n_rows + 1, 5,
        width_ratios=(1.4, 4.2, 1.3, 4.2, 1.3),
        height_ratios=(0.4,) + (1.0,) * n_rows,
        hspace=0.18, wspace=0.05,
    )

    # Column headers (in row 0)
    ax_h_label = fig.add_subplot(gs[0, 0])
    ax_h_label.axis("off")
    ax_h_ff = fig.add_subplot(gs[0, 1:3])
    ax_h_ff.axis("off")
    ax_h_ff.text(
        0.5, 0.0, "Worst false friend  (in top-k, large \u0394pKi)",
        transform=ax_h_ff.transAxes,
        fontsize=12, fontweight="bold", ha="center", va="bottom",
        color="#9c2d2d",
    )
    ax_h_catch = fig.add_subplot(gs[0, 3:5])
    ax_h_catch.axis("off")
    ax_h_catch.text(
        0.5, 0.0, "Best catch  (outside top-k, MCS-verified cliff)",
        transform=ax_h_catch.transAxes,
        fontsize=12, fontweight="bold", ha="center", va="bottom",
        color="#1f5c2e",
    )

    for ri, sid in enumerate(short_ids, start=1):
        group, color, _ = style_for(sid)

        # Label cell (column 0)
        ax_label = fig.add_subplot(gs[ri, 0])
        ax_label.axis("off")
        ff_ex = false_friends[sid]
        ax_label.text(
            0.05, 0.55, ff_ex.name,
            transform=ax_label.transAxes,
            fontsize=11, fontweight="bold", color=color,
            ha="left", va="center",
        )
        ax_label.text(
            0.05, 0.30, group,
            transform=ax_label.transAxes,
            fontsize=8.5, color=color, ha="left", va="center",
        )

        # Worst false friend (columns 1, 2)
        ax_ff_mol = fig.add_subplot(gs[ri, 1])
        ax_ff_metrics = fig.add_subplot(gs[ri, 2])
        ff_lines = [
            (f"|\u0394pKi| = {ff_ex.delta_y:.2f}", True),
            (f"({_format_fold(ff_ex.delta_y)} potency)", False),
            (f"FP similarity = {ff_ex.similarity:.2f}", False),
            (f"neighbor rank = {ff_ex.neighbor_rank}", False),
        ]
        _draw_pair_cell(
            ax_ff_mol, ax_ff_metrics,
            mols[ff_ex.i], mols[ff_ex.j],
            float(y[ff_ex.i]), float(y[ff_ex.j]),
            ff_lines,
        )

        # Best catch (columns 3, 4)
        ax_c_mol = fig.add_subplot(gs[ri, 3])
        ax_c_metrics = fig.add_subplot(gs[ri, 4])
        catch = best_catches[sid]
        if catch is None:
            _draw_no_catch_cell(ax_c_mol, ax_c_metrics)
        else:
            catch_lines = [
                (f"|\u0394pKi| = {catch.delta_y:.2f}", True),
                (f"({_format_fold(catch.delta_y)} potency)", False),
                (f"FP similarity = {catch.similarity:.2f}", False),
                (f"MCS overlap = {catch.mcs_fraction:.2f}", False),
            ]
            _draw_pair_cell(
                ax_c_mol, ax_c_metrics,
                mols[catch.i], mols[catch.j],
                float(y[catch.i]), float(y[catch.j]),
                catch_lines,
            )

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

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


# Keep the old single-column function for backwards compatibility / simpler use.
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
    """Single-column (worst-false-friend only) version of `plot_cliff_examples`."""
    short_ids = list(examples.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]
    n_rows = len(short_ids)

    fig_w = 12.0
    fig_h = row_height * n_rows + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows, 3, width_ratios=(1.6, 5.0, 1.4),
        hspace=0.15, wspace=0.05,
    )

    for ri, sid in enumerate(short_ids):
        ex = examples[sid]
        group, color, _ = style_for(sid)
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

        ax_mol = fig.add_subplot(gs[ri, 1])
        ax_metrics = fig.add_subplot(gs[ri, 2])
        lines = [
            (f"|\u0394pKi| = {ex.delta_y:.2f}", True),
            (f"({_format_fold(ex.delta_y)} potency)", False),
            (f"FP similarity = {ex.similarity:.2f}", False),
            (f"neighbor rank = {ex.neighbor_rank}", False),
        ]
        _draw_pair_cell(
            ax_mol, ax_metrics,
            mols[ex.i], mols[ex.j],
            float(y[ex.i]), float(y[ex.j]),
            lines,
        )

    if title:
        fig.suptitle(title, fontsize=13, y=0.995)

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
