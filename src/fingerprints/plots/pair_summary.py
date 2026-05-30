"""Summary plot: per-pair distance across all fingerprint methods.

Condenses the eight per-fingerprint similarity matrices to a single grouped
bar chart showing how each fingerprint scored each pair (A, B, C, D).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from loguru import logger
from matplotlib.patches import Patch
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D

from fingerprints.fingerprint_methods.base import FingerprintResult
from fingerprints.fingerprint_methods.similarity import pairwise_similarity
from fingerprints.plots.groupings import (
    GROUP_ORDER as _GROUP_ORDER,
    grouped_order as _grouped_order,
    style_for as _style_for,
)


def _draw_pair_image(
    mol_a: Mol,
    mol_b: Mol,
    label_a: str,
    label_b: str,
    size: tuple[int, int] = (600, 280),
) -> np.ndarray:
    """Render two molecules side by side into a single RGB image."""
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


def plot_pair_summary_bars(
    fps: dict[str, FingerprintResult],
    pair_boundaries: list[tuple[int, int]],
    pair_labels: list[str],
    out_path: Path,
    mols: list[Mol] | None = None,
    mol_labels: list[str] | None = None,
    figsize: tuple[float, float] = (18.0, 10.0),
    dpi: int = 200,
) -> Path:
    """Grouped bar chart of within-pair distance for each fingerprint, with
    an optional row of structure pairs as a header.

    Args:
        fps: dict from short id -> FingerprintResult
        pair_boundaries: list of (lo, hi) tuples specifying pair indices
        pair_labels: short label per pair (e.g. ["A: minor change", ...])
        out_path: file to write
        mols: optional list of all molecules in the matrix (rows of `fps`).
            When provided together with `mol_labels`, the figure adds a
            top row of structure pairs aligned with the bar groups.
        mol_labels: human-readable labels for `mols`
        figsize: matplotlib figure size in inches
        dpi: output DPI

    Notes:
        Similarities live on different scales by kind (Tanimoto in [0,1],
        cosine in [-1,1]). We plot distance = 1 - similarity so that
        "small bar = close pair" is consistent across both kinds. Distance
        for cosine ranges in [0, 2]; for Tanimoto in [0, 1].
    """
    n_fps = len(fps)
    n_pairs = len(pair_boundaries)

    # Reorder fps so members of the same group sit next to each other.
    short_ids = list(fps.keys())
    order = _grouped_order(short_ids)
    short_ids = [short_ids[i] for i in order]
    ordered_fps = [fps[sid] for sid in short_ids]

    similarities = np.zeros((n_fps, n_pairs))
    display_names: list[str] = []
    for fi, fp in enumerate(ordered_fps):
        sim = pairwise_similarity(fp)
        for pi, (lo, hi) in enumerate(pair_boundaries):
            similarities[fi, pi] = sim[lo, hi]
        display_names.append(fp.name)

    show_structures = mols is not None and mol_labels is not None
    if show_structures:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, n_pairs, height_ratios=[1.0, 3.0], hspace=0.05)
        struct_axes = [fig.add_subplot(gs[0, c]) for c in range(n_pairs)]
        ax = fig.add_subplot(gs[1, :])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        struct_axes = []

    bar_width = 0.85 / n_fps
    x = np.arange(n_pairs)

    styles = [_style_for(sid, fi) for fi, sid in enumerate(short_ids)]

    for fi in range(n_fps):
        offsets = x + (fi - (n_fps - 1) / 2) * bar_width
        _, fill, hatch = styles[fi]
        ax.bar(
            offsets,
            similarities[fi],
            width=bar_width,
            color=fill,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
            label=display_names[fi],
        )
        for xi, s in zip(offsets, similarities[fi]):
            if s >= 0.92:
                # too close to the ceiling - put the label inside the bar
                y = s - 0.02
                va = "top"
                color = "white"
            elif s >= 0:
                y = s + 0.02
                va = "bottom"
                color = "black"
            else:
                y = s - 0.02
                va = "top"
                color = "black"
            ax.text(
                xi,
                y,
                f"{s:.2f}",
                ha="center",
                va=va,
                fontsize=8,
                rotation=90,
                color=color,
            )

    # baseline at zero in case any cosine values go negative
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=14)
    ax.set_xlabel("pair", fontsize=13)
    ax.set_ylabel("within-pair similarity", fontsize=13)
    y_lo = min(-0.05, similarities.min() - 0.15) if similarities.min() < 0 else 0.0
    ax.set_ylim(y_lo, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Build a legend that visually mirrors the grouping: one column per group,
    # with a bold group header followed by its member fingerprints.
    legend_handles: list = []
    legend_labels: list[str] = []
    by_group: dict[str, list[int]] = {g: [] for g in _GROUP_ORDER}
    for fi, (group, _, _) in enumerate(styles):
        by_group.setdefault(group, []).append(fi)

    # Pad columns to equal length so matplotlib's column-major fill aligns.
    max_col = max(len(v) + 1 for v in by_group.values())  # +1 for header
    columns: list[list[tuple]] = []
    for group in _GROUP_ORDER:
        members = by_group.get(group, [])
        col: list[tuple] = []
        # group header (use mathtext-safe header without breaking spaces)
        header = group.replace(" ", r"\ ")
        col.append((Patch(facecolor="none", edgecolor="none"), f"$\\bf{{{header}}}$"))
        for fi in members:
            _, fill, hatch = styles[fi]
            col.append(
                (
                    Patch(facecolor=fill, edgecolor="black", hatch=hatch),
                    display_names[fi],
                )
            )
        # pad
        while len(col) < max_col:
            col.append((Patch(facecolor="none", edgecolor="none"), ""))
        columns.append(col)

    # Flatten column-major into a list (matplotlib fills row-major by default,
    # so we transpose: emit row 0 of every column, then row 1, ...).
    n_cols = len(columns)
    for ri in range(max_col):
        for ci in range(n_cols):
            handle, label = columns[ci][ri]
            legend_handles.append(handle)
            legend_labels.append(label)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        fontsize=9,
        ncol=n_cols,
        framealpha=0.95,
        handlelength=1.6,
        columnspacing=1.2,
    )

    # Structure header row aligned 1-to-1 with bar groups.
    if show_structures:
        assert mols is not None and mol_labels is not None
        for ci, (lo, hi) in enumerate(pair_boundaries):
            sax = struct_axes[ci]
            img = _draw_pair_image(
                mols[lo],
                mols[hi],
                mol_labels[lo],
                mol_labels[hi],
            )
            sax.imshow(img)
            sax.set_title(
                f"{mol_labels[lo]}    vs    {mol_labels[hi]}",
                fontsize=10,
            )
            sax.axis("off")

    fig.suptitle("Within-pair similarity by fingerprint", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path
