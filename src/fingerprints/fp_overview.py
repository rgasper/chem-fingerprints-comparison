"""Compute and summarize all fingerprints for a *single* molecule.

The notebook's core interaction is: pick one molecule upstream, then show how
each fingerprint represents it downstream. This module does the per-molecule
work — computing every classical fingerprint and packaging enough detail to
drive side-by-side visualizations (bit density, the raw bit vector, and a short
description of what each fingerprint encodes).

Neural fingerprints (CheMeleon, MIST) are intentionally excluded here: they load
torch/transformers and are slow, which would break the "flip between molecules
instantly" feel. They belong in precomputed sections, not the live selector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from fingerprints.fingerprint_methods.rdkit_fps import all_classical


# One-line description of what each classical fingerprint actually encodes,
# keyed by the short id used in `all_classical`.
FP_DESCRIPTIONS: dict[str, str] = {
    "morgan": "Circular atom environments (radius 2) hashed into bits — 'what does the neighborhood around each atom look like?'",
    "rdkit_topo": "Hashed linear paths through the molecular graph up to a max length.",
    "atom_pair": "All pairs of atoms encoded by their types and the topological distance between them.",
    "top_torsion": "Every 4-atom linear path (torsion), encoded by the atom types along it.",
    "maccs": "166 predefined yes/no structural keys — a human-readable substructure checklist.",
    "avalon": "Avalon toolkit's path- and feature-based hashed fingerprint (compact, 512 bits).",
}

# Preferred display order (simple -> complex).
FP_ORDER: tuple[str, ...] = (
    "maccs",
    "morgan",
    "avalon",
    "atom_pair",
    "top_torsion",
    "rdkit_topo",
)


@dataclass(frozen=True)
class FPSummary:
    key: str
    name: str
    kind: str
    n_features: int
    n_on: int
    density: float  # n_on / n_features
    on_indices: tuple[int, ...]
    description: str


def summarize_all(mol: Chem.Mol) -> list[FPSummary]:
    """Return an FPSummary for each classical fingerprint of this molecule."""
    results = all_classical([mol])
    out: list[FPSummary] = []
    for key in FP_ORDER:
        r = results[key]
        vec = r.array[0]
        on = tuple(int(i) for i in np.nonzero(vec)[0])
        out.append(
            FPSummary(
                key=key,
                name=r.name,
                kind=r.kind,
                n_features=r.n_features,
                n_on=len(on),
                density=len(on) / r.n_features if r.n_features else 0.0,
                on_indices=on,
                description=FP_DESCRIPTIONS.get(key, ""),
            )
        )
    return out
