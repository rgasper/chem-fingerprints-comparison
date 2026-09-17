"""Read the cached kNN cliff-failure analysis (``scripts/analyze_knn_cliffs.py``)
for the notebook's "why a similarity model can't see the cliff" section.

kNN regression on ECFP is the simplest model whose behaviour *is* the
fingerprint's similarity: a molecule's prediction is the mean activity of its
Tanimoto-nearest neighbours. So its blindness to a cliff is the fingerprint's
blindness. This module just serves the precomputed numbers; it never trains.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CACHE = Path("data/knn_cliffs/knn_cliffs.json")


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(CACHE.read_text())


def has_data() -> bool:
    return CACHE.exists()


def endpoints() -> list[str]:
    return list(_data()["endpoints"].keys()) if has_data() else []


def k_grid() -> list[int]:
    return list(_data()["k_grid"]) if has_data() else []


def k_curve(endpoint: str) -> list[dict]:
    """[{k, r2}, ...] held-out R^2 vs neighbourhood size for an endpoint."""
    return _data()["endpoints"][endpoint]["k_curve"]


def endpoint_meta(endpoint: str) -> dict:
    ep = _data()["endpoints"][endpoint]
    return {"n_total": ep["n_total"], "n_train": ep["n_train"], "n_test": ep["n_test"]}


def cliff_pair(endpoint: str, index: int) -> dict | None:
    """The analysis record for one curated cliff pair, or None if not present.

    Shape: {index, cliff_on, change, mol1, mol2} where each mol is
    {smiles, true, neighbors:[{smiles,tanimoto,activity}], pred_by_k:[{k,pred}]}.
    """
    if not has_data() or endpoint not in _data()["endpoints"]:
        return None
    for p in _data()["endpoints"][endpoint]["cliff_pairs"]:
        if p["index"] == index:
            return p
    return None


def best_k(endpoint: str) -> dict:
    """The k that maximises held-out R^2 (the model's honest 'best setting')."""
    curve = k_curve(endpoint)
    return max(curve, key=lambda d: d["r2"])


def pred_at_k(mol_report: dict, k: int) -> float:
    """A molecule's kNN prediction at neighbourhood size k (nearest grid point)."""
    rows = mol_report["pred_by_k"]
    exact = next((r for r in rows if r["k"] == k), None)
    if exact is not None:
        return float(exact["pred"])
    # fall back to the closest available k on the grid
    closest = min(rows, key=lambda r: abs(r["k"] - k))
    return float(closest["pred"])
