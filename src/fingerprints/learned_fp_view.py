"""Read the cached learned-fingerprint alpha-grid results for the notebook.

``scripts/train_alpha_grid.py`` trains the D-MPNN across a grid of loss weights
(alpha) offline and caches, per alpha: test R^2 and RMSE on both endpoints
(mean/std over seeds) plus per-test-molecule predicted/actual/cliff for the
scatter. This module just loads that JSON, so the notebook stays light (no
torch/chemprop import) and instant.
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


def list_endpoints() -> list[dict]:
    """Every single endpoint we have a learned model for, as flat records:
    ``{label, pair_key, side}`` where side is 'a' or 'b'.

    Lets the notebook offer a plain endpoint picker without exposing the
    two-task training detail.
    """
    out: list[dict] = []
    for pk in available_pairs():
        g = load_grid(pk)
        out.append({"label": g["target_a"], "pair_key": pk, "side": "a"})
        out.append({"label": g["target_b"], "pair_key": pk, "side": "b"})
    return out


def predicted_vs_measured(pair_key: str, side: str) -> dict:
    """Held-out (test-split) predicted vs. measured pKi for one endpoint, read
    off the balanced (middle-alpha) cached model.

    Returns ``{label, actual, pred, rmse, r2, n}`` with NaN/unlabeled rows
    already dropped. Never raises for a present grid.
    """
    g = load_grid(pair_key)
    label = g["target_a"] if side == "a" else g["target_b"]
    results = g["results"]
    row = results[len(results) // 2]  # balanced model
    sc = row.get("scatter") or {}
    actual = sc.get("actual_" + side, [])
    pred = sc.get("pred_" + side, [])
    is_test = sc.get("is_test", [1] * len(actual))
    xs: list[float] = []
    ys: list[float] = []
    for a, p, t in zip(actual, pred, is_test):
        if not t:
            continue
        if a is None or p is None or a != a or p != p:
            continue
        xs.append(float(a))
        ys.append(float(p))
    n = len(xs)
    if n:
        rmse = (sum((a - b) ** 2 for a, b in zip(xs, ys)) / n) ** 0.5
        mean = sum(xs) / n
        ss_tot = sum((a - mean) ** 2 for a in xs)
        ss_res = sum((a - b) ** 2 for a, b in zip(xs, ys))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        rmse = float("nan")
        r2 = float("nan")
    return {"label": label, "actual": xs, "pred": ys, "rmse": rmse, "r2": r2, "n": n}
