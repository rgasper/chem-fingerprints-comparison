"""Alignment plots: kNN sweep curves and final bar charts.

Two plot types:
- `plot_knn_sweep`: per-fingerprint R\u00b2 (or ROC-AUC) vs k as line curves
  for one property. Used to pick a sensible single k.
- `plot_alignment_bars`: bar chart with two metrics per fingerprint (kNN
  at chosen k + Spearman \u03c1) for one property. Group-colored, hatched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Patch
from typeguard import typechecked

from fingerprints.clustering.alignment import AlignmentResult
from fingerprints.plots.groupings import (
    GROUP_ORDER,
    grouped_order,
    style_for,
)


@typechecked
def plot_knn_sweep(
    alignments: dict[str, AlignmentResult],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 7.0),
    dpi: int = 200,
) -> Path:
    """One line per fingerprint, x = k, y = kNN score (R\u00b2 or ROC-AUC)."""
    short_ids = list(alignments.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    for sid in short_ids:
        a = alignments[sid]
        group, color, hatch = style_for(sid)
        ax.plot(
            a.ks,
            a.knn_scores,
            marker="o",
            label=a.name,
            color=color,
            linewidth=2.0,
            markersize=6,
        )

    ax.set_xscale("log")
    ax.set_xticks(alignments[short_ids[0]].ks)
    ax.set_xticklabels([str(k) for k in alignments[short_ids[0]].ks])
    ax.set_xlabel("k (nearest neighbors)", fontsize=12)
    score_name = (
        "R\u00b2"
        if alignments[short_ids[0]].task_type == "regression"
        else "ROC-AUC"
    )
    ax.set_ylabel(f"leave-one-out kNN {score_name}", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.95)
    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


@typechecked
def plot_alignment_bars(
    alignments: dict[str, AlignmentResult],
    k_chosen: int,
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (16.0, 8.0),
    dpi: int = 200,
) -> Path:
    """Two-panel bar chart: left panel = kNN at k_chosen, right = Spearman \u03c1.

    Bars ordered by group, colored + hatched per fingerprint matching the
    other figures.
    """
    short_ids = list(alignments.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    n_fps = len(short_ids)
    knn_vals = np.zeros(n_fps)
    spearman_vals = np.zeros(n_fps)
    display_names: list[str] = []
    styles: list[tuple[str, str, str]] = []

    for fi, sid in enumerate(short_ids):
        a = alignments[sid]
        if k_chosen not in a.ks:
            raise ValueError(
                f"k={k_chosen} not in swept ks={a.ks}; rerun the sweep with "
                f"k_chosen included"
            )
        ki = a.ks.index(k_chosen)
        knn_vals[fi] = a.knn_scores[ki]
        spearman_vals[fi] = a.spearman_rho
        display_names.append(a.name)
        styles.append(style_for(sid, fi))

    score_name = (
        "R\u00b2"
        if alignments[short_ids[0]].task_type == "regression"
        else "ROC-AUC"
    )

    fig, (ax_knn, ax_sp) = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(n_fps)
    width = 0.7

    panels = (
        (
            ax_knn, knn_vals,
            f"kNN {score_name} (k={k_chosen})",
            "LOCAL alignment\n(does the property change smoothly\nwith fingerprint distance?)",
        ),
        (
            ax_sp, spearman_vals,
            "Spearman \u03c1 (distance, |\u0394y|)",
            "GLOBAL alignment\n(do far-apart molecules in fingerprint\nspace differ more in property?)",
        ),
    )
    for ax, vals, ylabel, subtitle in panels:
        for fi, (group, color, hatch) in enumerate(styles):
            ax.bar(
                x[fi], vals[fi],
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch,
            )
            # Value label above (or below if negative)
            v = vals[fi]
            if v >= 0:
                ax.text(
                    x[fi], v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, rotation=0,
                )
            else:
                ax.text(
                    x[fi], v - 0.01, f"{v:.2f}",
                    ha="center", va="top", fontsize=9, rotation=0,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(subtitle, fontsize=11, color="#444444", loc="left")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Symmetric y-limits where possible: kNN R\u00b2 in [-..1]; ROC-AUC in [0,1];
    # Spearman in [-1,1]. Let kNN auto-scale but cap top at 1.
    if alignments[short_ids[0]].task_type == "regression":
        ax_knn.set_ylim(min(-0.05, knn_vals.min() - 0.1), 1.0)
    else:
        ax_knn.set_ylim(0.4, 1.0)
    sp_min = float(min(-0.05, spearman_vals.min() - 0.1))
    sp_max = float(max(0.05, spearman_vals.max() + 0.1, 1.0))
    ax_sp.set_ylim(sp_min, sp_max)

    # Build a shared legend underneath for the four groups
    legend_handles = []
    legend_labels = []
    seen_groups = set()
    for fi, (group, color, hatch) in enumerate(styles):
        if group in seen_groups:
            continue
        seen_groups.add(group)
        legend_handles.append(
            Patch(facecolor=color, edgecolor="black", hatch="", label=group)
        )
        legend_labels.append(group)
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10,
    )

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout(rect=(0, 0.03, 1.0, 0.96 if title else 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
