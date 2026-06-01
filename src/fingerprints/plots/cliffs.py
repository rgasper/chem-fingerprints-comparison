"""Plots for graph-distance-defined activity-cliff analysis.

Three figure types:

- `plot_cliff_similarity_violins`: distribution of FP similarities over
  cliff pairs, one violin per fingerprint. Lower violins = better cliff
  resolution. The headline figure.
- `plot_cliff_examples_per_fp`: per-fingerprint deep-dive showing top-3
  most cliff-blind and top-3 most cliff-aware molecule pairs across all
  three datasets. One image file per fingerprint.
- `plot_cliff_blind_summary`: cross-dataset heatmap of P(sim >= 0.7)
  per fingerprint per dataset.
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
    CliffExamplePair,
    CliffSimResult,
    mcs_diff_atoms,
)
from fingerprints.plots.groupings import grouped_order, style_for


def _draw_pair_image(
    mol_a: Mol,
    mol_b: Mol,
    highlight_a: list[int] | None = None,
    highlight_b: list[int] | None = None,
    size: tuple[int, int] = (700, 260),
    highlight_color: tuple[float, float, float] = (1.0, 0.55, 0.55),  # soft red
) -> np.ndarray:
    """Render two molecules side by side, optionally highlighting atoms.

    If `highlight_a` / `highlight_b` are given, the listed atom indices
    on each molecule are drawn with a red highlight. Used to mark atoms
    outside the maximum common substructure - i.e. the parts that
    differ between the pair.
    """
    half_w = size[0] // 2
    h = size[1]

    def _draw_one(mol: Mol, highlight: list[int] | None) -> Image.Image:
        drawer = rdMolDraw2D.MolDraw2DCairo(half_w, h)
        drawer.drawOptions().clearBackground = True
        if highlight:
            atom_colors = {idx: highlight_color for idx in highlight}
            # Also highlight bonds connecting two highlight atoms
            highlight_bonds: list[int] = []
            bond_colors: dict[int, tuple[float, float, float]] = {}
            highlight_set = set(highlight)
            for bond in mol.GetBonds():
                if (bond.GetBeginAtomIdx() in highlight_set
                        and bond.GetEndAtomIdx() in highlight_set):
                    highlight_bonds.append(bond.GetIdx())
                    bond_colors[bond.GetIdx()] = highlight_color
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight,
                highlightAtomColors=atom_colors,
                highlightBonds=highlight_bonds,
                highlightBondColors=bond_colors,
            )
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")

    img_a = _draw_one(mol_a, highlight_a)
    img_b = _draw_one(mol_b, highlight_b)
    composite = Image.new("RGB", size, "white")
    composite.paste(img_a, (0, 0))
    composite.paste(img_b, (half_w, 0))
    return np.asarray(composite)


def _format_fold(delta_y: float) -> str:
    fold = 10 ** delta_y
    if fold >= 1000:
        return f"{fold/1000:.0f},000\u00d7"
    return f"{fold:.0f}\u00d7"


@typechecked
def plot_cliff_similarity_violins(
    results: dict[str, CliffSimResult],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (13.0, 6.5),
    dpi: int = 200,
) -> Path:
    """Distribution of cliff-pair similarities per fingerprint.

    Each violin shows the distribution of FP similarity scores across
    activity cliff pairs (graph-distance-defined cliffs, no FP involved
    in the cliff definition). Lower = better cliff resolution. The
    horizontal dashed line at 0.7 marks a conventional Tanimoto
    "very similar" threshold; bars / violins above it represent
    "cliff-blind" cases.
    """
    short_ids = list(results.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    n = len(short_ids)
    data: list[np.ndarray] = []
    names: list[str] = []
    styles: list[tuple[str, str, str]] = []
    medians: list[float] = []
    for i, sid in enumerate(short_ids):
        r = results[sid]
        data.append(r.sims)
        names.append(r.name)
        styles.append(style_for(sid, i))
        medians.append(r.median_sim)

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(
        data, positions=np.arange(n),
        showmeans=False, showmedians=True, showextrema=False,
        widths=0.85,
    )
    for body, (group, color, hatch) in zip(parts["bodies"], styles):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.85)
        body.set_linewidth(0.6)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

    # Annotate the median above each violin
    for xi, med in enumerate(medians):
        ax.text(
            xi, med + 0.04, f"{med:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(
        0.7, color="red", linestyle="--", linewidth=1.0,
        label="cliff-blind threshold (sim \u2265 0.7)",
    )

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("fingerprint similarity over cliff pairs", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    if title:
        ax.set_title(title, fontsize=13)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


@typechecked
def plot_cliff_examples_per_fp(
    short_id: str,
    fp_display_name: str,
    examples_by_dataset: dict[str, tuple[list[CliffExamplePair], list[CliffExamplePair]]],
    mols_by_dataset: dict[str, list[Mol]],
    y_by_dataset: dict[str, np.ndarray],
    out_path: Path,
    title: str = "",
    row_height: float = 2.6,
    dpi: int = 200,
) -> Path:
    """Per-fingerprint deep-dive example figure.

    Layout: one row per dataset, six mol-pair columns per row. The first
    three columns are the FP's top-3 most cliff-blind pairs (highest
    similarity on a true cliff); the last three are its top-3 most
    cliff-aware (lowest similarity on a true cliff). Each panel is a
    side-by-side molecule pair with non-MCS atoms highlighted, pKi values
    underneath, and the FP similarity / |Delta pKi| / graph distance
    above.

    Args:
        short_id: fingerprint short id (used only for color coding here).
        fp_display_name: human-readable label, e.g. "Morgan(r=2,2048b)".
        examples_by_dataset: dict mapping dataset label to
            (most_blind_pairs, most_aware_pairs); each list is length 3
            (or shorter if too few cliff pairs in that dataset).
        mols_by_dataset: dict mapping dataset label to its molecule list.
        y_by_dataset: dict mapping dataset label to its pKi array.
        out_path: png destination.
        title: figure title.
    """
    datasets = list(examples_by_dataset.keys())
    n_rows = len(datasets)
    if n_rows == 0:
        raise ValueError("examples_by_dataset is empty")

    group, color, _ = style_for(short_id)

    # 3 most-blind columns + visual gap + 3 most-aware columns
    n_blind_cols = 3
    n_aware_cols = 3
    col_count = 1 + n_blind_cols + n_aware_cols  # leftmost is dataset label

    # Width ratios: label narrow, then 6 wide mol panels with a small gap
    width_ratios = [0.6] + [1.0] * n_blind_cols + [1.0] * n_aware_cols

    fig_w = sum(width_ratios) * 1.95
    fig_h = row_height * n_rows + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        n_rows + 1, col_count,
        width_ratios=width_ratios,
        height_ratios=(0.5,) + (1.0,) * n_rows,
        hspace=0.45, wspace=0.10,
    )

    # Header row
    ax_h_label = fig.add_subplot(gs[0, 0])
    ax_h_label.axis("off")
    ax_h_blind = fig.add_subplot(gs[0, 1:1 + n_blind_cols])
    ax_h_blind.axis("off")
    ax_h_blind.text(
        0.5, 0.0,
        "Most cliff-blind  (highest similarity on true cliffs)",
        transform=ax_h_blind.transAxes,
        fontsize=12, fontweight="bold", ha="center", va="bottom",
        color="#9c2d2d",
    )
    ax_h_aware = fig.add_subplot(gs[0, 1 + n_blind_cols:])
    ax_h_aware.axis("off")
    ax_h_aware.text(
        0.5, 0.0,
        "Most cliff-aware  (lowest similarity on true cliffs)",
        transform=ax_h_aware.transAxes,
        fontsize=12, fontweight="bold", ha="center", va="bottom",
        color="#1f5c2e",
    )

    for ri, ds in enumerate(datasets, start=1):
        most_blind, most_aware = examples_by_dataset[ds]
        mols = mols_by_dataset[ds]
        y = y_by_dataset[ds]

        # Dataset label column
        ax_label = fig.add_subplot(gs[ri, 0])
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.5, ds,
            transform=ax_label.transAxes,
            fontsize=11, fontweight="bold", color="black",
            ha="center", va="center", wrap=True,
        )

        # Most cliff-blind columns
        for ci in range(n_blind_cols):
            ax = fig.add_subplot(gs[ri, 1 + ci])
            if ci < len(most_blind):
                _draw_example_panel(ax, most_blind[ci], mols, y)
            else:
                _draw_no_example_panel(ax)

        # Most cliff-aware columns
        for ci in range(n_aware_cols):
            ax = fig.add_subplot(gs[ri, 1 + n_blind_cols + ci])
            if ci < len(most_aware):
                _draw_example_panel(ax, most_aware[ci], mols, y)
            else:
                _draw_no_example_panel(ax)

    if title:
        fig.suptitle(title, fontsize=14, y=0.995, color=color, fontweight="bold")

    fig.subplots_adjust(
        top=0.94 if title else 0.99,
        bottom=0.02,
        left=0.005, right=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


def _draw_example_panel(
    ax,
    ex: CliffExamplePair,
    mols: list[Mol],
    y: np.ndarray,
) -> None:
    """Draw one mol-pair panel (single Axes) with metric annotations on top."""
    diff = mcs_diff_atoms(mols[ex.i], mols[ex.j])
    hl_a, hl_b = (diff if diff is not None else (None, None))
    img = _draw_pair_image(
        mols[ex.i], mols[ex.j],
        highlight_a=hl_a, highlight_b=hl_b,
    )
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.5)

    # Metrics above the structures, two lines so they don't run together
    fold = _format_fold(ex.delta_y)
    line1 = f"sim = {ex.similarity:.2f}"
    line2 = f"|\u0394pKi| = {ex.delta_y:.2f} ({fold}),  gd = {ex.graph_distance}"
    ax.text(
        0.5, 1.16, line1,
        transform=ax.transAxes,
        fontsize=10, ha="center", va="bottom",
        fontweight="bold",
    )
    ax.text(
        0.5, 1.02, line2,
        transform=ax.transAxes,
        fontsize=8.5, ha="center", va="bottom",
        color="#444444",
    )

    # pKi labels under the two molecules
    ax.text(
        0.25, -0.04, f"pKi={float(y[ex.i]):.2f}",
        transform=ax.transAxes,
        fontsize=8.5, ha="center", va="top",
    )
    ax.text(
        0.75, -0.04, f"pKi={float(y[ex.j]):.2f}",
        transform=ax.transAxes,
        fontsize=8.5, ha="center", va="top",
    )


def _draw_no_example_panel(ax) -> None:
    """Placeholder when a dataset has fewer cliff pairs than n_top."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f5f5f5")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.5)
    ax.text(
        0.5, 0.5, "(none)",
        transform=ax.transAxes,
        fontsize=10, ha="center", va="center",
        color="#888888", style="italic",
    )


@typechecked
def plot_cliff_blind_summary(
    results_by_dataset: dict[str, dict[str, CliffSimResult]],
    out_path: Path,
    threshold: float = 0.7,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 4.5),
    dpi: int = 200,
) -> Path:
    """Heatmap summary: rows = datasets, cols = fingerprints, cells =
    fraction of cliff pairs scored above `threshold` similarity.

    Higher = more cliff-blind = worse. Magma colormap (inverted) so
    high cliff-blindness shows as bright/light cells.
    """
    datasets = list(results_by_dataset.keys())
    if not datasets:
        raise ValueError("results_by_dataset is empty")

    first = results_by_dataset[datasets[0]]
    short_ids = list(first.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    mat = np.zeros((len(datasets), len(short_ids)))
    for ri, ds in enumerate(datasets):
        for ci, sid in enumerate(short_ids):
            r = results_by_dataset[ds][sid]
            mat[ri, ci] = (
                r.cliff_blind_rate_07 if threshold == 0.7
                else r.cliff_blind_rate_05 if threshold == 0.5
                else float((r.sims >= threshold).mean())
            )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mat, cmap="magma_r", aspect="auto",
        vmin=0.0, vmax=1.0,
    )
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            color = "white" if v > 0.55 else "black"
            ax.text(
                ci, ri, f"{v:.2f}",
                ha="center", va="center", fontsize=10, color=color,
            )

    ax.set_xticks(np.arange(len(short_ids)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(
        [first[sid].name for sid in short_ids],
        rotation=35, ha="right", fontsize=9,
    )
    yticklabels = [
        f"{ds}\n(n={results_by_dataset[ds][short_ids[0]].n_pairs} cliffs)"
        for ds in datasets
    ]
    ax.set_yticklabels(yticklabels, fontsize=9)
    for tick, sid in zip(ax.get_xticklabels(), short_ids):
        _, color, _ = style_for(sid)
        tick.set_color(color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(
        f"cliff-blind rate: P(similarity \u2265 {threshold})  (lower = better)",
        fontsize=10,
    )

    if title:
        ax.set_title(title, fontsize=13, pad=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
