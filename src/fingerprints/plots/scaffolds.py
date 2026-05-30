"""Scaffold-purity plots: bar chart of purity-at-k, plus property/purity Pareto.

Two figures:
- `plot_scaffold_purity_bars`: per-fingerprint scaffold purity at chosen k,
  with the random-match baseline drawn as a horizontal line.
- `plot_purity_property_pareto`: scatter of (scaffold purity, property kNN
  R\u00b2 / ROC-AUC) at the same k, one panel per ADME property. Reads as a
  Pareto plot \u2014 points to the upper-left are the most useful (high property
  alignment, low scaffold dependence).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Patch
from typeguard import typechecked

from fingerprints.clustering.alignment import AlignmentResult
from fingerprints.clustering.scaffolds import ScaffoldPurityResult
from fingerprints.plots.groupings import grouped_order, style_for


@typechecked
def plot_scaffold_purity_bars(
    purities: dict[str, ScaffoldPurityResult],
    k: int,
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 6.5),
    dpi: int = 200,
) -> Path:
    """Bar chart of scaffold purity at k for each fingerprint.

    A dashed horizontal line shows the random-match baseline (the value
    purity would take for a fingerprint that ignores scaffold).
    """
    short_ids = list(purities.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    n = len(short_ids)
    vals = np.zeros(n)
    names: list[str] = []
    styles: list[tuple[str, str, str]] = []
    baseline = 0.0
    for i, sid in enumerate(short_ids):
        p = purities[sid]
        if k not in p.ks:
            raise ValueError(f"k={k} not in {p.ks}")
        ki = p.ks.index(k)
        vals[i] = p.purity_at_k[ki]
        names.append(p.name)
        styles.append(style_for(sid, i))
        baseline = p.baseline  # same across fingerprints (depends only on data)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n)
    width = 0.7
    for i, (group, color, hatch) in enumerate(styles):
        ax.bar(
            x[i], vals[i],
            width=width, color=color, edgecolor="black",
            linewidth=0.6, hatch=hatch,
        )
        ax.text(
            x[i], vals[i] + 0.005, f"{vals[i]:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(
        baseline, color="black", linestyle="--", linewidth=1.0,
        label=f"random baseline = {baseline:.4f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(f"scaffold purity at k={k}", fontsize=12)
    # Auto-scale to data so the bars are readable. ChEMBL is so scaffold-
    # diverse that a fixed [0, 1] y-axis flattens the differences.
    ymax = max(float(vals.max()), baseline) * 1.4
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    # Group legend below
    legend_handles, legend_labels, seen = [], [], set()
    for fi, (group, color, hatch) in enumerate(styles):
        if group in seen:
            continue
        seen.add(group)
        legend_handles.append(
            Patch(facecolor=color, edgecolor="black", label=group)
        )
        legend_labels.append(group)
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10,
    )

    if title:
        ax.set_title(title, fontsize=13)

    fig.tight_layout(rect=(0, 0.03, 1.0, 0.97 if title else 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


@typechecked
def plot_purity_property_pareto(
    purities_by_property: dict[str, dict[str, ScaffoldPurityResult]],
    alignments_by_property: dict[str, dict[str, AlignmentResult]],
    k: int,
    out_path: Path,
    title: str = (
        "Tradeoff: does the fingerprint hop scaffolds while staying property-aware?"
    ),
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
) -> Path:
    """Pareto scatter: scaffold purity (x) vs property kNN score (y).

    One panel per property. Axes are oriented so 'better' is up-and-to-the-
    left: high property alignment with low scaffold dependence.

    Both metrics are property-specific because both depend on which
    molecules are in the dataset (different ADME datasets have different
    molecule populations, hence different scaffold distributions).

    Args:
        purities_by_property: dict property_name -> dict(short_id -> ScaffoldPurityResult)
        alignments_by_property: dict property_name -> dict(short_id -> AlignmentResult)
        k: shared k value for both metrics; must be present in every result
        out_path: png destination
    """
    properties = list(alignments_by_property.keys())
    if list(purities_by_property.keys()) != properties:
        raise ValueError(
            "purities_by_property and alignments_by_property must have the "
            "same keys in the same order"
        )
    n_props = len(properties)
    if figsize is None:
        figsize = (6.0 * n_props, 5.5)
    fig, axes = plt.subplots(1, n_props, figsize=figsize, sharey=False)
    if n_props == 1:
        axes = [axes]

    for pi, prop in enumerate(properties):
        ax = axes[pi]
        aligns = alignments_by_property[prop]
        purities = purities_by_property[prop]
        baseline = next(iter(purities.values())).baseline

        short_ids = list(aligns.keys())
        order = grouped_order(short_ids)
        short_ids = [short_ids[i] for i in order]

        score_label = "R\u00b2"
        for sid in short_ids:
            a = aligns[sid]
            p = purities[sid]
            if k not in a.ks:
                raise ValueError(f"k={k} not in alignment ks={a.ks}")
            if k not in p.ks:
                raise ValueError(f"k={k} not in purity ks={p.ks}")
            ki_a = a.ks.index(k)
            ki_p = p.ks.index(k)
            x = p.purity_at_k[ki_p]
            y = a.knn_scores[ki_a]
            score_label = (
                "R\u00b2" if a.task_type == "regression" else "ROC-AUC"
            )
            _, color, _ = style_for(sid)
            ax.scatter(
                x, y,
                s=180, color=color, edgecolor="black", linewidth=0.7,
                marker="o", zorder=3,
            )
            ax.annotate(
                a.name, (x, y),
                xytext=(7, 4), textcoords="offset points",
                fontsize=8, color=color,
            )

        ax.axvline(
            baseline, color="black", linestyle="--", linewidth=0.8,
            label=f"random baseline = {baseline:.3f}",
        )
        ax.set_xlabel(f"scaffold purity at k={k}", fontsize=11)
        ax.set_ylabel(f"property kNN {score_label} at k={k}", fontsize=11)
        ax.set_title(prop, fontsize=12)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout(rect=(0, 0.0, 1.0, 0.94 if title else 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
