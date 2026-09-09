"""Precompute context-dependent activity cliffs across pairs of related targets.

A *context-dependent cliff* is a pair of near-identical molecules whose potency
gap is large on one target but small on another. It's the cleanest possible
demonstration that "activity cliff" is not a property of a molecular pair alone
- it depends on which endpoint you ask about. That contextual nature is a big
part of why cliffs are hard to model.

This search is an all-pairs MCS over the molecules shared between two targets,
which is far too slow to run interactively (minutes). So we run it here once,
rank candidates by how lopsided the two endpoints are, and dump the top
candidates to JSON for hand-vetting. The notebook then loads a small curated
subset instantly.

Usage:
    uv run python scripts/precompute_context_cliffs.py

Output:
    data/context_cliffs_raw.json  - top candidates per target pair, for review
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS, rdFingerprintGenerator

from fingerprints.data.molace import ALL_MOLACE_DATASETS, load_molace


CACHE_MOLACE = Path(".cache/molace")
OUT_PATH = Path("data/context_cliffs_raw.json")

# Target pairs to search: closely-related targets where selectivity is a real
# medicinal-chemistry goal, chosen for high shared-molecule overlap.
TARGET_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("CHEMBL234_Ki", "Dopamine D3", "CHEMBL219_Ki", "Dopamine D4"),
    ("CHEMBL233_Ki", "mu-opioid", "CHEMBL237_Ki", "kappa-opioid"),
)

# Cliff thresholds (match the repo's graph-distance cliff definition).
GRAPH_DIST_MAX = 3
BIG_GAP = 2.0   # >= ~100x potency difference = a cliff
FLAT_GAP = 1.0  # < ~10x = essentially flat
TANIMOTO_PREFILTER = 0.6  # only graph-close pairs survive; keeps MCS load sane
N_KEEP = 40  # top candidates per target pair to write out


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _canon(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def _load_target(name: str) -> dict[str, float]:
    ds = next(d for d in ALL_MOLACE_DATASETS if d.name == name)
    df = load_molace(ds, CACHE_MOLACE)
    out: dict[str, float] = {}
    for smiles, y in zip(df["smiles"].to_list(), df["y"].to_list()):
        c = _canon(smiles)
        if c is not None and y is not None:
            out[c] = float(y)  # last write wins; MoleculeACE is deduplicated
    return out


def _graph_distance(mol_i: Chem.Mol, mol_j: Chem.Mol) -> int | None:
    res = rdFMCS.FindMCS(
        [mol_i, mol_j],
        timeout=2,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        completeRingsOnly=False,
        ringMatchesRingOnly=True,
    )
    if res.canceled:
        return None
    return mol_i.GetNumHeavyAtoms() + mol_j.GetNumHeavyAtoms() - 2 * res.numAtoms


def search_pair(
    name_a: str, label_a: str, name_b: str, label_b: str
) -> list[dict]:
    da, db = _load_target(name_a), _load_target(name_b)
    shared = sorted(set(da) & set(db))
    logger.info(f"{label_a} x {label_b}: {len(shared)} shared molecules")

    mols = {s: Chem.MolFromSmiles(s) for s in shared}
    fps = {s: _MORGAN.GetFingerprint(mols[s]) for s in shared}
    n_atoms = {s: mols[s].GetNumHeavyAtoms() for s in shared}

    found: list[dict] = []
    for s1, s2 in itertools.combinations(shared, 2):
        if abs(n_atoms[s1] - n_atoms[s2]) > GRAPH_DIST_MAX:
            continue
        if DataStructs.TanimotoSimilarity(fps[s1], fps[s2]) < TANIMOTO_PREFILTER:
            continue
        d_a = abs(da[s1] - da[s2])
        d_b = abs(db[s1] - db[s2])
        if not (max(d_a, d_b) >= BIG_GAP and min(d_a, d_b) < FLAT_GAP):
            continue
        gd = _graph_distance(mols[s1], mols[s2])
        if gd is None or not (1 <= gd <= GRAPH_DIST_MAX):
            continue
        cliff_on = label_a if d_a >= d_b else label_b
        found.append(
            {
                "target_a": {"name": name_a, "label": label_a},
                "target_b": {"name": name_b, "label": label_b},
                "smiles_1": s1,
                "smiles_2": s2,
                "graph_distance": gd,
                "delta_a": round(d_a, 2),
                "delta_b": round(d_b, 2),
                "y1_a": round(da[s1], 2),
                "y2_a": round(da[s2], 2),
                "y1_b": round(db[s1], 2),
                "y2_b": round(db[s2], 2),
                "cliff_on": cliff_on,
                "contrast": round(abs(d_a - d_b), 2),
            }
        )
    found.sort(key=lambda r: -r["contrast"])
    logger.info(f"  -> {len(found)} context-dependent cliffs")
    return found[:N_KEEP]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}
    for name_a, label_a, name_b, label_b in TARGET_PAIRS:
        key = f"{label_a} vs {label_b}"
        out[key] = search_pair(name_a, label_a, name_b, label_b)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    logger.info(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
