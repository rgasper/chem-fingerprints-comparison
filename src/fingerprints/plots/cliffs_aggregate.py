"""Plots for matched-control cliff separation analysis.

Three figure types complement the existing cliff figures:

- `plot_pr_auc_summary`: heatmap of PR-AUC per (dataset, fingerprint),
  same layout as `plot_cliff_blind_summary`. Higher = better separation
  of cliffs from matched non-cliff controls.
- `plot_metric_scatter`: per-target scatter of cliff-blind rate (x) vs
  PR-AUC (y), one point per (dataset, FP). Visualizes the "scale vs
  ranking" decoupling: a point in the upper-right has the FP both
  scoring cliffs as high-similarity AND failing to rank cliffs below
  matched non-cliffs; a point in the upper-left has the FP scoring
  cliffs as low-similarity (good cliff-blind rate) AND ranking them
  correctly. Neural FPs typically land top-right under cosine.
- `plot_neural_metric_compare`: for the two neural FPs, side-by-side
  paired distributions of cliff vs matched-non-cliff under cosine and
  L2. Shows whether the cliff-blindness story changes when you switch
  metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from typeguard import typechecked

from fingerprints.clustering.cliffs_aggregate import CliffSeparationResult
from fingerprints.plots.groupings import grouped_order, style_for


@typechecked
def plot_pr_auc_summary(
    results_by_dataset: dict[str, dict[str, CliffSeparationResult]],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 4.5),
    dpi: int = 200,
) -> Path:
    """Heatmap: rows = datasets, cols = fingerprints, cells = PR-AUC of
    cliff vs matched-non-cliff separation.

    Higher = better. We use a sequential viridis colormap so high values
    are visibly distinct. The cell shows numeric PR-AUC; values near 0.5
    mean random separation given 1:1 prevalence.
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
            mat[ri, ci] = results_by_dataset[ds][sid].pr_auc

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            color = "white" if v < 0.75 else "black"
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
        f"{ds}\n(n={results_by_dataset[ds][short_ids[0]].n_cliffs} cliffs / "
        f"{results_by_dataset[ds][short_ids[0]].n_noncliffs} non-cliffs)"
        for ds in datasets
    ]
    ax.set_yticklabels(yticklabels, fontsize=9)
    for tick, sid in zip(ax.get_xticklabels(), short_ids):
        _, color, _ = style_for(sid)
        tick.set_color(color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(
        "PR-AUC: cliff vs matched-non-cliff (higher = better)",
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


@typechecked
def plot_metric_scatter(
    cliff_blind_by_dataset: dict[str, dict[str, float]],
    pr_auc_by_dataset: dict[str, dict[str, float]],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (10.0, 7.0),
    dpi: int = 200,
) -> Path:
    """Scatter cliff-blind-rate (x) vs PR-AUC (y), one panel per dataset.

    Each (dataset, FP) pair contributes one marker. Color = FP family,
    marker = FP. The x axis is the absolute-threshold metric (lower =
    better cliff awareness on a fixed cutoff); the y axis is the
    rank-based metric (higher = better cliff vs non-cliff separation).

    The point of the figure is to show that the two metrics disagree
    for some FPs - in particular, neural FPs typically land far right
    on the x axis (high cliff-blind rate) but their y position varies,
    showing whether "cliff-blind" is a scale problem (high y) or a real
    inability to rank (low y).
    """
    datasets = list(cliff_blind_by_dataset.keys())
    if not datasets:
        raise ValueError("cliff_blind_by_dataset is empty")
    if set(datasets) != set(pr_auc_by_dataset.keys()):
        raise ValueError("dataset key sets disagree between inputs")

    n_panels = len(datasets)
    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize, sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    short_ids = list(cliff_blind_by_dataset[datasets[0]].keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    for panel_idx, (ax, ds) in enumerate(zip(axes, datasets)):
        for sid in short_ids:
            x = cliff_blind_by_dataset[ds][sid]
            y = pr_auc_by_dataset[ds][sid]
            _, color, _ = style_for(sid)
            ax.scatter(
                [x], [y],
                s=100, c=[color],
                edgecolor="black", linewidth=0.6, zorder=3,
            )
            ax.annotate(
                sid,
                xy=(x, y),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=color,
            )

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, zorder=1,
                   label="random baseline")
        ax.set_xlim(-0.05, 1.05)
        # Auto y-limits with a floor below 0.5 so worse-than-random points
        # are visible. Some FPs (notably MIST cosine on D3) actively rank
        # cliffs as MORE similar than non-cliffs, giving PR-AUC < 0.5.
        all_y = [
            pr_auc_by_dataset[d][s]
            for d in datasets for s in short_ids
        ]
        y_lo = min(0.45, min(all_y) - 0.03)
        y_hi = max(1.02, max(all_y) + 0.03)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("cliff-blind rate at sim \u2265 0.7\n(lower = better)", fontsize=10)
        if panel_idx == 0:
            ax.set_ylabel("PR-AUC of cliff vs matched non-cliff\n(higher = better)", fontsize=10)
        ax.set_title(ds, fontsize=11)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


@typechecked
def plot_neural_metric_compare(
    cosine_results_by_dataset: dict[str, dict[str, CliffSeparationResult]],
    l2_results_by_dataset: dict[str, dict[str, CliffSeparationResult]],
    neural_short_ids: list[str],
    out_path: Path,
    title: str = "",
    dpi: int = 200,
) -> Path:
    """For each neural FP, paired cliff vs non-cliff distributions
    under cosine and L2.

    Layout: rows = neural FPs, columns = datasets, each cell shows two
    pairs of split violins (cosine cliff/non-cliff, L2 cliff/non-cliff).
    PR-AUC for each metric is annotated on the cell.

    Reading: if cosine and L2 give similar PR-AUC, the cliff-blindness
    is real - the FP genuinely can't rank cliffs vs matched non-cliffs.
    If L2 PR-AUC is much higher than cosine, the cliff-blindness was a
    metric artifact and L2 recovers the discrimination.
    """
    datasets = list(cosine_results_by_dataset.keys())
    if not datasets:
        raise ValueError("cosine_results_by_dataset is empty")

    n_rows = len(neural_short_ids)
    n_cols = len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    for row_idx, sid in enumerate(neural_short_ids):
        for col_idx, ds in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            cos = cosine_results_by_dataset[ds][sid]
            l2 = l2_results_by_dataset[ds][sid]

            # x positions: 1, 2 for cosine cliff/non-cliff; 4, 5 for L2.
            positions = [1, 2, 4, 5]
            data = [cos.cliff_sims, cos.noncliff_sims, l2.cliff_sims, l2.noncliff_sims]
            parts = ax.violinplot(
                data, positions=positions, showmedians=True, widths=0.85,
            )
            colors = ["#d95f02", "#1b9e77", "#d95f02", "#1b9e77"]
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.6)
                pc.set_edgecolor("black")

            ax.set_xticks([1.5, 4.5])
            ax.set_xticklabels(["cosine\n(1 - sim)", "L2\n(1 - dist)"], fontsize=9)

            # PR-AUC labels at the panel top, fixed in axes coordinates.
            ax.text(
                1.5, 1.04, f"PR-AUC = {cos.pr_auc:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="lightgray",
                          boxstyle="round,pad=0.25", alpha=0.95),
            )
            ax.text(
                4.5, 1.04, f"PR-AUC = {l2.pr_auc:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="lightgray",
                          boxstyle="round,pad=0.25", alpha=0.95),
            )

            if row_idx == 0:
                ax.set_title(ds, fontsize=11, pad=22)
            if col_idx == 0:
                _, color, _ = style_for(sid)
                ax.set_ylabel(
                    f"{cos.name}\n1 - distance", fontsize=10, color=color,
                )
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Single legend
    cliff_patch = plt.Rectangle((0, 0), 1, 1, fc="#d95f02", alpha=0.6, ec="black")
    nc_patch = plt.Rectangle((0, 0), 1, 1, fc="#1b9e77", alpha=0.6, ec="black")
    fig.legend(
        [cliff_patch, nc_patch],
        ["cliff pairs (should be low)", "matched non-cliff pairs (should be high)"],
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, frameon=False, fontsize=10,
    )

    if title:
        fig.suptitle(title, fontsize=13, y=1.05)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
