"""Heatmap of pairwise RV coefficients between fingerprints.

A single k x k square heatmap. Rows/cols are reordered to match the
project's GROUP_ORDER (Local atom-env -> Path-based -> Expert keys -> Neural)
so the block structure is visible. Group boundaries are drawn as black lines
and group labels are placed on the top axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from typeguard import typechecked

from fingerprints.clustering.agreement import AgreementResult
from fingerprints.plots.groupings import GROUPS, grouped_order, style_for


@typechecked
def plot_rv_heatmap(
    result: AgreementResult,
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (9.0, 8.0),
    dpi: int = 200,
    annotate: bool = True,
) -> Path:
    """Render the RV-coefficient matrix as a heatmap.

    Args:
        result: output of `rv_matrix`.
        out_path: png destination.
        title: figure title.
        figsize: matplotlib figure size in inches.
        dpi: output dpi.
        annotate: whether to write the numeric value in each cell.
    """
    order = grouped_order(result.short_ids)
    sids = [result.short_ids[i] for i in order]
    names = [result.display_names[i] for i in order]
    rv = result.rv[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=figsize)
    # RV is non-negative; clip floor to 0 for display (numerical noise can
    # produce tiny negatives).
    im = ax.imshow(
        np.clip(rv, 0.0, 1.0),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )

    # Diagonal text in light color, off-diagonal in dark
    if annotate:
        for i in range(rv.shape[0]):
            for j in range(rv.shape[1]):
                v = rv[i, j]
                color = "white" if v < 0.55 else "black"
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=9, color=color,
                )

    ax.set_xticks(np.arange(len(sids)))
    ax.set_yticks(np.arange(len(sids)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    # Group boundary lines: draw a thin black line between cells whose group
    # changes. Compute group per ordered fp.
    groups = [GROUPS[s][0] if s in GROUPS else "Other" for s in sids]
    boundaries = [
        i for i in range(1, len(groups)) if groups[i] != groups[i - 1]
    ]
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=1.2)
        ax.axvline(b - 0.5, color="black", linewidth=1.2)

    # Color the y-tick labels by group, matching the rest of the figures
    for tick, sid in zip(ax.get_yticklabels(), sids):
        _, color, _ = style_for(sid)
        tick.set_color(color)
    for tick, sid in zip(ax.get_xticklabels(), sids):
        _, color, _ = style_for(sid)
        tick.set_color(color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("RV coefficient", fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, pad=12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
