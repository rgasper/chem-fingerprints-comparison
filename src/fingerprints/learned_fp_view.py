"""Read the cached learned-fingerprint alpha-grid results for the notebook.

``scripts/train_alpha_grid.py`` trains the D-MPNN across a grid of loss weights
(alpha) offline and caches, per alpha: test R^2 on both endpoints (mean/std over
seeds) and a 2D PCA of the learned fingerprint. This module just loads that
JSON, so the notebook stays light (no torch/chemprop import) and instant.
"""

from __future__ import annotations

import json
from pathlib import Path

GRID_DIR = Path("data/learned_fp")


def has_grid(pair_key: str) -> bool:
    return (GRID_DIR / f"{pair_key}_alpha_grid.json").exists()


def load_grid(pair_key: str) -> dict:
    """Full cached grid for an endpoint pair (raises if not yet trained)."""
    return json.loads((GRID_DIR / f"{pair_key}_alpha_grid.json").read_text())


def available_pairs() -> list[str]:
    return sorted(p.stem.replace("_alpha_grid", "") for p in GRID_DIR.glob("*_alpha_grid.json"))
