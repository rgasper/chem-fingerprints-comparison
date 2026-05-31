"""Plots for the activity-cliff probes.

Three figure types per dataset:

- `plot_false_friend_bars`: false-friend rate at top-k per fingerprint.
  Headline figure. Lower bars = fingerprint's neighborhoods have fewer
  cliff pairs hiding in them.
- `plot_neighbor_dy_violins`: distribution of |Delta y| across each
  fingerprint's top-k neighbor pairs. Reveals the tail behavior - a
  fingerprint with a low rate but a heavy upper tail is still hosting
  some big cliffs.
- `plot_cliff_rmse_bars`: kNN regression RMSE on cliff vs non-cliff test
  molecules, with the penalty (cliff - non-cliff) annotated. Mirrors the
  MoleculeACE benchmark paper's evaluation.

Plus one cross-dataset summary:

- `plot_false_friend_summary`: heatmap of false-friend rate
  (datasets x fingerprints).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Patch
from typeguard import typechecked

from fingerprints.clustering.cliffs import (
    CatchRateResult,
    CliffRMSEResult,
    FalseFriendResult,
)
from fingerprints.plots.groupings import grouped_order, style_for


@typechecked
def plot_false_friend_bars(
    results: dict[str, FalseFriendResult],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 6.5),
    dpi: int = 200,
) -> Path:
    """Bar chart of false-friend rate per fingerprint.

    A 'false friend' is a top-k neighbor pair with |Delta y| above the
    cliff threshold. Lower is better.
    """
    short_ids = list(results.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    n = len(short_ids)
    vals = np.zeros(n)
    names: list[str] = []
    styles: list[tuple[str, str, str]] = []
    k = next(iter(results.values())).k
    threshold = next(iter(results.values())).cliff_threshold
    for i, sid in enumerate(short_ids):
        r = results[sid]
        vals[i] = r.false_friend_rate
        names.append(r.name)
        styles.append(style_for(sid, i))

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

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(
        f"false-friend rate at top-k={k}\n(fraction of neighbors with |\u0394y| \u2265 {threshold})",
        fontsize=11,
    )
    ymax = max(float(vals.max()) * 1.25, 0.05)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

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
def plot_neighbor_dy_violins(
    results: dict[str, FalseFriendResult],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (13.0, 6.5),
    dpi: int = 200,
) -> Path:
    """Violin plot of |Delta y| distributions across each fingerprint's
    top-k neighbor pairs.

    The cliff threshold is drawn as a horizontal dashed line; portions of
    the violin above it are the false-friend region. A short, bottom-
    weighted violin = the fingerprint's neighborhoods are tight in
    activity. A heavy upper tail = it harbors cliffs.
    """
    short_ids = list(results.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    n = len(short_ids)
    data: list[np.ndarray] = []
    names: list[str] = []
    styles: list[tuple[str, str, str]] = []
    threshold = next(iter(results.values())).cliff_threshold
    k = next(iter(results.values())).k
    for i, sid in enumerate(short_ids):
        r = results[sid]
        data.append(r.delta_y)
        names.append(r.name)
        styles.append(style_for(sid, i))

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

    ax.axhline(
        threshold, color="red", linestyle="--", linewidth=1.2,
        label=f"cliff threshold = {threshold} ({int(10**threshold)}\u00d7 potency)",
    )

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(f"|\u0394y| within top-k={k} neighbors (pKi units)", fontsize=11)
    ax.set_ylim(0, max(threshold * 3.5, float(np.percentile(np.concatenate(data), 99)) * 1.1))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    if title:
        ax.set_title(title, fontsize=13)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


@typechecked
def plot_cliff_rmse_bars(
    results: dict[str, CliffRMSEResult],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (13.0, 6.5),
    dpi: int = 200,
) -> Path:
    """Grouped bar chart: rmse_noncliff vs rmse_cliff per fingerprint.

    The penalty (cliff - non-cliff) is annotated above each pair.
    """
    short_ids = list(results.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    n = len(short_ids)
    rmse_nc = np.zeros(n)
    rmse_c = np.zeros(n)
    names: list[str] = []
    styles: list[tuple[str, str, str]] = []
    k = next(iter(results.values())).k
    for i, sid in enumerate(short_ids):
        r = results[sid]
        rmse_nc[i] = r.rmse_noncliff
        rmse_c[i] = r.rmse_cliff
        names.append(r.name)
        styles.append(style_for(sid, i))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n)
    width = 0.38
    # Two bars per fp: lighter = non-cliff, full color = cliff.
    for i, (group, color, hatch) in enumerate(styles):
        ax.bar(
            x[i] - width / 2, rmse_nc[i], width=width,
            color=color, alpha=0.45, edgecolor="black", linewidth=0.6,
        )
        ax.bar(
            x[i] + width / 2, rmse_c[i], width=width,
            color=color, edgecolor="black", linewidth=0.6, hatch=hatch,
        )
        # Penalty (signed) above the cliff bar
        penalty = rmse_c[i] - rmse_nc[i]
        sign = "+" if penalty >= 0 else "\u2212"
        y_top = max(rmse_nc[i], rmse_c[i])
        ax.text(
            x[i], y_top + 0.03, f"\u0394 {sign}{abs(penalty):.2f}",
            ha="center", va="bottom", fontsize=8, color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(f"kNN regression RMSE on test set (k={k})", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Custom legend explaining bar pairs + group colors
    bar_legend = [
        Patch(facecolor="#888888", alpha=0.45, edgecolor="black", label="non-cliff test mols"),
        Patch(facecolor="#888888", edgecolor="black", label="cliff test mols"),
    ]
    seen, group_legend = set(), []
    for fi, (group, color, hatch) in enumerate(styles):
        if group in seen:
            continue
        seen.add(group)
        group_legend.append(
            Patch(facecolor=color, edgecolor="black", label=group)
        )

    ax.legend(handles=bar_legend, loc="upper left", fontsize=9, framealpha=0.95)
    fig.legend(
        handles=group_legend,
        loc="lower center", ncol=len(group_legend),
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
def plot_false_friend_summary(
    rates_by_dataset: dict[str, dict[str, FalseFriendResult]],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 4.5),
    dpi: int = 200,
) -> Path:
    """Heatmap summary: rows = datasets, cols = fingerprints.

    Cells show false-friend rate. Lower (cooler) is better. Useful as a
    one-glance summary across the three datasets.
    """
    datasets = list(rates_by_dataset.keys())
    if not datasets:
        raise ValueError("rates_by_dataset is empty")

    # Use the column order from the first dataset (all should share it)
    first = rates_by_dataset[datasets[0]]
    short_ids = list(first.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    mat = np.zeros((len(datasets), len(short_ids)))
    for ri, ds in enumerate(datasets):
        for ci, sid in enumerate(short_ids):
            mat[ri, ci] = rates_by_dataset[ds][sid].false_friend_rate

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mat, cmap="magma_r", aspect="auto",
        vmin=0.0, vmax=float(mat.max()) * 1.05,
    )
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            color = "white" if v > 0.55 * mat.max() else "black"
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
    ax.set_yticklabels(datasets, fontsize=10)
    for tick, sid in zip(ax.get_xticklabels(), short_ids):
        _, color, _ = style_for(sid)
        tick.set_color(color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("false-friend rate (lower = better)", fontsize=10)

    if title:
        ax.set_title(title, fontsize=13, pad=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


@typechecked
def plot_catch_rate_summary(
    rates_by_dataset: dict[str, dict[str, CatchRateResult]],
    out_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 4.5),
    dpi: int = 200,
) -> Path:
    """Heatmap summary of cliff catch rate: rows = datasets, cols = fingerprints.

    Catch rate = fraction of MCS-verified cliff candidates the fingerprint
    correctly placed outside both molecules' top-k neighbor sets. Higher
    (warmer) is better - the dual of the false-friend rate.
    """
    datasets = list(rates_by_dataset.keys())
    if not datasets:
        raise ValueError("rates_by_dataset is empty")

    first = rates_by_dataset[datasets[0]]
    short_ids = list(first.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    mat = np.zeros((len(datasets), len(short_ids)))
    for ri, ds in enumerate(datasets):
        for ci, sid in enumerate(short_ids):
            mat[ri, ci] = rates_by_dataset[ds][sid].catch_rate

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mat, cmap="magma", aspect="auto",
        vmin=0.0, vmax=1.0,
    )
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            color = "white" if v < 0.5 else "black"
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
        f"{ds}\n(n={next(iter(rates_by_dataset[ds].values())).n_candidates} cliffs)"
        for ds in datasets
    ]
    ax.set_yticklabels(yticklabels, fontsize=9)
    for tick, sid in zip(ax.get_xticklabels(), short_ids):
        _, color, _ = style_for(sid)
        tick.set_color(color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("catch rate (higher = better)", fontsize=10)

    if title:
        ax.set_title(title, fontsize=13, pad=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
