"""Shared fingerprint groupings used by every multi-fingerprint figure.

Four conceptual families:
- Local atom-environment / circular  (Morgan)
- Path-based topological              (RDKit-topo, Avalon, AtomPair, TopTorsion)
- Predefined expert keys              (MACCS)
- Learned neural embeddings           (CheMeleon, MIST-28M, MIST-1.8B)

Each member gets a unique (color, hatch) pair, with shared hue per group.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


# (group label, fill color, hatch pattern)
GROUPS: dict[str, tuple[str, str, str]] = {
    # Local atom-environment / circular - blues
    "morgan":      ("Local atom-env",  "#2E5BBA", ""),
    # Path / topological hashed - greens
    "rdkit_topo":  ("Path-based",      "#3F7A35", ""),
    "avalon":      ("Path-based",      "#7DB873", "\\\\"),
    "atom_pair":   ("Path-based",      "#A8D49E", "//"),
    "top_torsion": ("Path-based",      "#C8E4BD", ".."),
    # Predefined expert keys - orange
    "maccs":       ("Expert keys",     "#F58518", ""),
    # Learned neural embeddings - purples
    "chemeleon":   ("Neural",          "#6B3F9E", ""),
    "mist_28M":    ("Neural",          "#9467BD", "xx"),
    "mist_1_8B":   ("Neural",          "#B898D6", "++"),
}

GROUP_ORDER: list[str] = ["Local atom-env", "Path-based", "Expert keys", "Neural"]


def style_for(short_id: str, fallback_idx: int = 0) -> tuple[str, str, str]:
    """Return (group label, fill color, hatch pattern) for a short fp id.

    Falls back to tab10 with no group for unknown ids.
    """
    if short_id in GROUPS:
        return GROUPS[short_id]
    return "Other", plt.get_cmap("tab10")(fallback_idx % 10), ""


def grouped_order(short_ids: list[str]) -> list[int]:
    """Return permutation indices that order short_ids by GROUP_ORDER, then by
    insertion order of GROUPS within each group. Unknown ids go last."""
    rank: dict[str, tuple[int, int]] = {}
    for i, sid in enumerate(GROUPS):
        group = GROUPS[sid][0]
        rank[sid] = (GROUP_ORDER.index(group), i)
    fallback = (len(GROUP_ORDER), 999)
    return sorted(
        range(len(short_ids)),
        key=lambda i: rank.get(short_ids[i], (fallback[0], fallback[1] + i)),
    )
