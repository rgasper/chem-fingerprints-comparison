"""2x4 panel plotter for per-fingerprint UMAP figures (clustering + ADME).

One panel per fingerprint, ordered by group, with a small colored frame on
each panel indicating the fingerprint's group. Supports either categorical
coloring (e.g. HDBSCAN cluster id) or continuous coloring (e.g. logS).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import Normalize, TwoSlopeNorm
from typeguard import typechecked

from fingerprints.clustering.embed import EmbeddingResult
from fingerprints.plots.groupings import GROUP_ORDER, grouped_order, style_for


@typechecked
def plot_umap_grid(
    embeddings: dict[str, EmbeddingResult],
    out_path: Path,
    color_values: np.ndarray | None = None,
    color_kind: str = "cluster",  # "cluster" | "continuous" | "binary" | "none"
    color_label: str = "",
    cmap: str = "viridis",
    cmap_diverging_center: float | None = None,
    title: str = "",
    point_size: float = 4.0,
    point_alpha: float = 0.55,
    figsize: tuple[float, float] = (24.0, 12.0),
    dpi: int = 200,
) -> Path:
    """Render a 2x4 grid of UMAP scatter panels, one per fingerprint.

    Args:
        embeddings: dict short_id -> EmbeddingResult
        out_path: file to write
        color_values: per-molecule values to color points by. When color_kind
            is "cluster", this is ignored and each panel is colored by its
            own HDBSCAN labels (each panel uses its own cluster ids since
            the clusterings are independent).
        color_kind: how to color points
            - "cluster": use each panel's own HDBSCAN labels (categorical
                tab20 palette per panel; -1 = grey for noise)
            - "continuous": use color_values with cmap (and optional
                cmap_diverging_center for TwoSlopeNorm)
            - "binary": use color_values \\in {0,1} with two distinct colors
            - "none": all points one color
        color_label: text for the colorbar (continuous case)
        cmap: matplotlib colormap name (continuous / binary)
        cmap_diverging_center: if set, use a diverging-style normalization
            with white at this value
        title: figure suptitle
        point_size: scatter dot size
        point_alpha: scatter alpha
        figsize: matplotlib figure size
        dpi: output DPI

    Returns:
        Path that was written
    """
    short_ids = list(embeddings.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    n = len(short_ids)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    # When using a continuous colorbar, reserve a thin extra column on the
    # right for it via gridspec so it doesn't compete with tight_layout.
    needs_cbar = color_kind == "continuous" and color_values is not None
    if needs_cbar:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            n_rows, n_cols + 1,
            width_ratios=[1.0] * n_cols + [0.04],
            wspace=0.08, hspace=0.18,
        )
        axes_flat = [fig.add_subplot(gs[r, c])
                     for r in range(n_rows) for c in range(n_cols)]
        cbar_ax = fig.add_subplot(gs[:, n_cols])
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, squeeze=False
        )
        axes_flat = list(axes.flatten())
        cbar_ax = None

    # set up the colormap normalization once for continuous coloring so all
    # 8 panels share the same scale
    norm = None
    if color_kind == "continuous" and color_values is not None:
        finite = color_values[np.isfinite(color_values)]
        if cmap_diverging_center is not None and finite.size:
            vmin, vmax = float(finite.min()), float(finite.max())
            norm = TwoSlopeNorm(
                vmin=min(vmin, cmap_diverging_center - 1e-6),
                vcenter=cmap_diverging_center,
                vmax=max(vmax, cmap_diverging_center + 1e-6),
            )
        elif finite.size:
            norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))

    # binary colormap for "binary" mode
    binary_colors = ("#cccccc", "#d62728")  # 0 = grey, 1 = red

    last_continuous_sc = None
    for ax_idx, sid in enumerate(short_ids):
        ax = axes_flat[ax_idx]
        emb = embeddings[sid]
        coords = emb.coords
        group, fill_color, _ = style_for(sid, ax_idx)

        # Color the panel border to indicate group; keep title dark for
        # readability and just underline-color it via a small group tag.
        for spine in ax.spines.values():
            spine.set_edgecolor(fill_color)
            spine.set_linewidth(2.5)

        if color_kind == "cluster":
            labels = emb.cluster_labels
            if labels is None:
                # fall back to all-one-color
                ax.scatter(coords[:, 0], coords[:, 1], s=point_size,
                           alpha=point_alpha, c="#666666", linewidths=0)
            else:
                # noise points first (grey, lower alpha) so they sit under
                # the actual clusters
                noise_mask = labels == -1
                if noise_mask.any():
                    ax.scatter(
                        coords[noise_mask, 0], coords[noise_mask, 1],
                        s=point_size * 0.7, alpha=0.25, c="#bbbbbb",
                        linewidths=0,
                    )
                clustered_mask = ~noise_mask
                if clustered_mask.any():
                    palette = plt.get_cmap("tab20")
                    cluster_colors = palette(
                        (labels[clustered_mask] % 20) / 20.0
                    )
                    ax.scatter(
                        coords[clustered_mask, 0],
                        coords[clustered_mask, 1],
                        s=point_size,
                        alpha=point_alpha,
                        c=cluster_colors,
                        linewidths=0,
                    )
                # cluster count annotation
                ax.text(
                    0.02, 0.98,
                    f"{emb.n_clusters} clusters",
                    transform=ax.transAxes,
                    fontsize=9, va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                              pad=2),
                )
        elif color_kind == "continuous":
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                s=point_size, alpha=point_alpha,
                c=color_values, cmap=cmap, norm=norm, linewidths=0,
            )
            last_continuous_sc = sc
        elif color_kind == "binary":
            assert color_values is not None
            # Plot majority class first, then minority on top so the smaller
            # class isn't hidden under the larger. Same alpha for both so
            # the visual density reflects actual class proportions.
            counts = {0: int((color_values == 0).sum()),
                      1: int((color_values == 1).sum())}
            majority = max(counts, key=lambda k: counts[k])
            minority = 1 - majority
            for cls in (majority, minority):
                mask = color_values == cls
                if mask.any():
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1],
                        s=point_size,
                        alpha=point_alpha,
                        c=binary_colors[cls],
                        linewidths=0,
                    )
        else:  # "none"
            ax.scatter(
                coords[:, 0], coords[:, 1],
                s=point_size, alpha=point_alpha,
                c="#3a6db0", linewidths=0,
            )

        # Title in black for readability; use the group's color only for the
        # colored border. The group tag (under the name) is also kept black
        # since the colored border already conveys group identity.
        ax.set_title(f"{emb.name}\n[{group}]", fontsize=11, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False)

    # hide any unused axes
    for ax_idx in range(n, n_rows * n_cols):
        axes_flat[ax_idx].axis("off")

    if needs_cbar and last_continuous_sc is not None:
        assert cbar_ax is not None
        fig.colorbar(last_continuous_sc, cax=cbar_ax, label=color_label)
    elif color_kind == "binary":
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=binary_colors[0], markersize=8,
                   label="negative"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=binary_colors[1], markersize=8,
                   label="positive"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=11)

    if title:
        fig.suptitle(title, fontsize=15)

    # Skip tight_layout when we have a manually-placed gridspec colorbar -
    # it tries to relayout in ways that break the colorbar placement.
    if not needs_cbar:
        bottom = 0.05 if color_kind == "binary" else 0
        top = 0.96 if title else 1.0
        fig.tight_layout(rect=(0, bottom, 1.0, top))
    elif title:
        # adjust top margin to leave room for the title
        fig.subplots_adjust(top=0.93)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


@typechecked
def plot_umap_grid_family_overlay(
    embeddings: dict[str, EmbeddingResult],
    out_path: Path,
    family_membership: np.ndarray,
    color_values: np.ndarray,
    color_kind: str,
    family_labels: list[str],
    color_label: str = "",
    cmap: str = "viridis",
    title: str = "",
    bg_point_size: float = 3.0,
    bg_point_alpha: float = 0.18,
    fg_point_size: float = 22.0,
    fg_point_alpha: float = 0.95,
    figsize: tuple[float, float] = (24.0, 12.0),
    dpi: int = 200,
) -> Path:
    """UMAP grid with all points greyed out and 1+ families overlaid.

    Args:
        embeddings: dict short_id -> EmbeddingResult (UMAP fit on full set)
        out_path: file to write
        family_membership: int array of shape (n_mols,). -1 = background,
            0..len(family_labels)-1 = family index. Points with family
            index >= 0 are drawn as larger markers on top, colored by
            color_values via cmap (continuous) or by binary palette.
        color_values: per-molecule property value
        color_kind: "continuous" or "binary"
        family_labels: human-readable labels for each family index, used in
            the legend (one entry per distinct family). Family 0 is drawn
            as circles, family 1 as triangles, etc. (up to 4 markers).
        color_label: text for the colorbar (continuous case)
        cmap: matplotlib colormap (continuous mode)
        title: figure suptitle

    Returns:
        Path that was written.
    """
    if color_kind not in ("continuous", "binary"):
        raise ValueError(f"unsupported color_kind={color_kind!r}")

    family_membership = np.asarray(family_membership)
    color_values = np.asarray(color_values)
    n_families = len(family_labels)
    if n_families == 0:
        raise ValueError("need at least one family")
    if n_families > 4:
        raise ValueError("only 4 distinct family markers supported")

    short_ids = list(embeddings.keys())
    order = grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]

    n = len(short_ids)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    needs_cbar = color_kind == "continuous"
    if needs_cbar:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            n_rows, n_cols + 1,
            width_ratios=[1.0] * n_cols + [0.04],
            wspace=0.08, hspace=0.18,
        )
        axes_flat = [fig.add_subplot(gs[r, c])
                     for r in range(n_rows) for c in range(n_cols)]
        cbar_ax = fig.add_subplot(gs[:, n_cols])
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, squeeze=False
        )
        axes_flat = list(axes.flatten())
        cbar_ax = None

    # shared color normalization across all panels
    norm = None
    if color_kind == "continuous":
        finite = color_values[np.isfinite(color_values)]
        if finite.size:
            norm = Normalize(vmin=float(finite.min()),
                             vmax=float(finite.max()))

    # one marker per family so they're distinguishable when overlapping
    family_markers = ["o", "^", "s", "D"][:n_families]
    binary_colors = ("#1f77b4", "#d62728")  # 0 = blue, 1 = red

    last_continuous_sc = None
    for ax_idx, sid in enumerate(short_ids):
        ax = axes_flat[ax_idx]
        emb = embeddings[sid]
        coords = emb.coords
        group, fill_color, _ = style_for(sid, ax_idx)

        for spine in ax.spines.values():
            spine.set_edgecolor(fill_color)
            spine.set_linewidth(2.5)

        # background: everything in light grey
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=bg_point_size, alpha=bg_point_alpha,
            c="#bbbbbb", linewidths=0,
        )

        # foreground: each family on top, colored by property
        for fi in range(n_families):
            mask = family_membership == fi
            if not mask.any():
                continue
            if color_kind == "continuous":
                sc = ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=fg_point_size, alpha=fg_point_alpha,
                    c=color_values[mask], cmap=cmap, norm=norm,
                    marker=family_markers[fi],
                    edgecolors="black", linewidths=0.4,
                )
                last_continuous_sc = sc
            else:  # binary
                vals = color_values[mask].astype(int)
                cols = np.array(
                    [binary_colors[v] for v in vals], dtype=object
                )
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=fg_point_size, alpha=fg_point_alpha,
                    c=list(cols),
                    marker=family_markers[fi],
                    edgecolors="black", linewidths=0.4,
                )

        ax.set_title(f"{emb.name}\n[{group}]", fontsize=11, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

    # hide any unused panels
    for ax_idx in range(n, n_rows * n_cols):
        axes_flat[ax_idx].axis("off")

    # colorbar / binary legend
    if needs_cbar and last_continuous_sc is not None:
        assert cbar_ax is not None
        fig.colorbar(last_continuous_sc, cax=cbar_ax, label=color_label)

    # family legend (marker shape encodes family identity)
    from matplotlib.lines import Line2D

    family_handles = [
        Line2D([0], [0], marker=family_markers[fi], color="w",
               markerfacecolor="#666666", markeredgecolor="black",
               markersize=9, label=family_labels[fi])
        for fi in range(n_families)
    ]
    if color_kind == "binary":
        binary_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=binary_colors[0], markersize=8,
                   label=f"{color_label} = 0"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=binary_colors[1], markersize=8,
                   label=f"{color_label} = 1"),
        ]
        handles = family_handles + binary_handles
    else:
        handles = family_handles

    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, 0.01),
               frameon=False, fontsize=11)

    if title:
        fig.suptitle(title, fontsize=15)

    if not needs_cbar:
        fig.tight_layout(rect=(0, 0.05, 1.0, 0.96 if title else 1.0))
    else:
        # leave room for the bottom legend and the top title
        fig.subplots_adjust(top=0.93 if title else 0.97, bottom=0.08)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
