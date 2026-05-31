"""Plots for graph-distance-defined activity-cliff analysis.

Three figure types:

- `plot_cliff_similarity_violins`: distribution of FP similarities over
  cliff pairs, one violin per fingerprint. Lower violins = better cliff
  resolution. The headline figure.
- `plot_cliff_examples`: per-fingerprint molecule-pair illustrations.
  Two columns: most cliff-blind pair (highest similarity) on the left,
  least cliff-blind (lowest similarity) on the right. Gives visual
  intuition for what each FP misses and catches.
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
)
from fingerprints.plots.groupings import grouped_order, style_for


def _draw_pair_image(
    mol_a: Mol,
    mol_b: Mol,
    size: tuple[int, int] = (700, 260),
) -> np.ndarray:
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


def _draw_pair_cell(
    ax_mol,
    ax_metrics,
    mol_a: Mol,
    mol_b: Mol,
    pki_a: float,
    pki_b: float,
    metric_lines: list[tuple[str, bool]],
) -> None:
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
        y_pos = 0.85 - li * (0.7 / max(n_lines - 1, 1))
        ax_metrics.text(
            0.0, y_pos, line,
            transform=ax_metrics.transAxes,
            fontsize=8.5, ha="left", va="center",
            fontweight="bold" if bold else "normal",
            color="black" if bold else "#444444",
        )


@typechecked
def plot_cliff_examples(
    examples: dict[str, tuple[CliffExamplePair, CliffExamplePair]],
    mols: list[Mol],
    y: np.ndarray,
    out_path: Path,
    title: str = "",
    row_height: float = 1.7,
    dpi: int = 200,
) -> Path:
    """8-row x 2-column figure: most cliff-blind pair on left, least on right.

    'Most cliff-blind' = the cliff pair this FP scored highest. 'Least
    cliff-blind' = the cliff pair this FP scored lowest. Both pairs are
    real activity cliffs from the dataset (graph-distance <= 5,
    |delta pKi| >= 2.0).
    """
    short_ids = list(examples.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]
    n_rows = len(short_ids)

    fig_w = 18.0
    fig_h = row_height * n_rows + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        n_rows + 1, 5,
        width_ratios=(1.4, 4.2, 1.3, 4.2, 1.3),
        height_ratios=(0.4,) + (1.0,) * n_rows,
        hspace=0.18, wspace=0.05,
    )

    # Headers
    ax_h_label = fig.add_subplot(gs[0, 0])
    ax_h_label.axis("off")
    ax_h_most = fig.add_subplot(gs[0, 1:3])
    ax_h_most.axis("off")
    ax_h_most.text(
        0.5, 0.0, "Most cliff-blind  (highest similarity on a true cliff)",
        transform=ax_h_most.transAxes,
        fontsize=12, fontweight="bold", ha="center", va="bottom",
        color="#9c2d2d",
    )
    ax_h_least = fig.add_subplot(gs[0, 3:5])
    ax_h_least.axis("off")
    ax_h_least.text(
        0.5, 0.0, "Least cliff-blind  (lowest similarity on a true cliff)",
        transform=ax_h_least.transAxes,
        fontsize=12, fontweight="bold", ha="center", va="bottom",
        color="#1f5c2e",
    )

    for ri, sid in enumerate(short_ids, start=1):
        most, least = examples[sid]
        group, color, _ = style_for(sid)

        ax_label = fig.add_subplot(gs[ri, 0])
        ax_label.axis("off")
        ax_label.text(
            0.05, 0.55, most.name,
            transform=ax_label.transAxes,
            fontsize=11, fontweight="bold", color=color,
            ha="left", va="center",
        )
        ax_label.text(
            0.05, 0.30, group,
            transform=ax_label.transAxes,
            fontsize=8.5, color=color, ha="left", va="center",
        )

        # Most cliff-blind
        ax_m_mol = fig.add_subplot(gs[ri, 1])
        ax_m_metrics = fig.add_subplot(gs[ri, 2])
        m_lines = [
            (f"FP similarity = {most.similarity:.2f}", True),
            (f"|\u0394pKi| = {most.delta_y:.2f}", False),
            (f"({_format_fold(most.delta_y)} potency)", False),
            (f"graph distance = {most.graph_distance}", False),
        ]
        _draw_pair_cell(
            ax_m_mol, ax_m_metrics,
            mols[most.i], mols[most.j],
            float(y[most.i]), float(y[most.j]),
            m_lines,
        )

        # Least cliff-blind
        ax_l_mol = fig.add_subplot(gs[ri, 3])
        ax_l_metrics = fig.add_subplot(gs[ri, 4])
        l_lines = [
            (f"FP similarity = {least.similarity:.2f}", True),
            (f"|\u0394pKi| = {least.delta_y:.2f}", False),
            (f"({_format_fold(least.delta_y)} potency)", False),
            (f"graph distance = {least.graph_distance}", False),
        ]
        _draw_pair_cell(
            ax_l_mol, ax_l_metrics,
            mols[least.i], mols[least.j],
            float(y[least.i]), float(y[least.j]),
            l_lines,
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
