"""Cross-fingerprint similarity scores for the 2 most cliff-blind ECFP examples.

For each dataset (D3, Thrombin, GSK3B):
1. Build ECFP (Morgan r=2) fingerprint.
2. Find the top-2 most cliff-blind pairs (highest Morgan Tanimoto similarity
   among all graph-distance-defined activity cliff pairs).
3. For those same molecule pairs, compute similarity under every other
   fingerprint: RDKit-topo, AtomPair, TopTorsion, MACCS, Avalon.
   (Neural FPs are skipped to keep this fast / reproducible without GPU.)

Run:
  python scripts/cliff_blind_cross_fp.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit.Chem import MolFromSmiles

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, "src")

from fingerprints.clustering.cliffs import (
    CliffPair,
    cliff_similarities,
    select_cliff_examples,
)
from fingerprints.data.molace import (
    D3_DOPAMINE,
    GSK3B,
    THROMBIN,
    MolACEDataset,
    load_molace,
)
from fingerprints.fingerprint_methods import rdkit_fps
from fingerprints.fingerprint_methods.chemeleon_fp import CheMeleonFingerprint
from fingerprints.fingerprint_methods.mist_fp import MIST_28M, MISTFingerprint

CACHE_MOLACE = Path(".cache/molace")
CACHE_CLIFFS = Path(".cache/cliff_pairs")

DATASETS = [D3_DOPAMINE, THROMBIN, GSK3B]

def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


_CHEMELEON: CheMeleonFingerprint | None = None
_MIST: MISTFingerprint | None = None


def _get_chemeleon() -> CheMeleonFingerprint:
    global _CHEMELEON
    if _CHEMELEON is None:
        _CHEMELEON = CheMeleonFingerprint(device=_device())
    return _CHEMELEON


def _get_mist() -> MISTFingerprint:
    global _MIST
    if _MIST is None:
        _MIST = MISTFingerprint(model_id=MIST_28M, device=_device())
    return _MIST


FP_BUILDERS = {
    "morgan":       lambda mols: rdkit_fps.morgan_fingerprint(mols),
    "rdkit_topo":   lambda mols: rdkit_fps.rdkit_topological_fingerprint(mols),
    "atom_pair":    lambda mols: rdkit_fps.atom_pair_fingerprint(mols),
    "top_torsion":  lambda mols: rdkit_fps.topological_torsion_fingerprint(mols),
    "maccs":        lambda mols: rdkit_fps.maccs_fingerprint(mols),
    "avalon":       lambda mols: rdkit_fps.avalon_fingerprint(mols),
    "chemeleon":    lambda mols: _get_chemeleon()(mols),
    "mist_28M":     lambda mols: _get_mist()(mols),
}


def _load_dataset(ds: MolACEDataset):
    df = load_molace(ds, CACHE_MOLACE)
    smis = df["smiles"].to_list()
    parsed = [(MolFromSmiles(s), i) for i, s in enumerate(smis)]
    keep = [(m, i) for m, i in parsed if m is not None]
    mols = [m for m, _ in keep]
    keep_idx = np.array([i for _, i in keep])
    y = df["y"].to_numpy()[keep_idx]
    smiles_list = [s for s, i in zip(smis, range(len(smis))) if (MolFromSmiles(s) is not None)]
    return mols, y, smiles_list


def _load_cliff_pairs(ds: MolACEDataset) -> list[CliffPair]:
    path = CACHE_CLIFFS / f"{ds.name}.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    a, b = fp_a.astype(bool), fp_b.astype(bool)
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return inter / union if union > 0 else 0.0


def main():
    for ds in DATASETS:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds.target_label} ({ds.name})")
        print(f"{'='*70}")

        mols, y, smiles_list = _load_dataset(ds)
        cliff_pairs = _load_cliff_pairs(ds)
        print(f"  Molecules: {len(mols)}, Cliff pairs: {len(cliff_pairs)}")

        # Build ECFP (Morgan) and select top-2 most cliff-blind pairs
        morgan_fp = FP_BUILDERS["morgan"](mols)
        most_blind, _ = select_cliff_examples(morgan_fp, cliff_pairs, n_top=2)

        # Build all other FPs once (shared across both examples)
        print(f"  Building all fingerprints...")
        all_fps = {sid: builder(mols) for sid, builder in FP_BUILDERS.items()}

        for rank, ex in enumerate(most_blind, start=1):
            i, j = ex.i, ex.j
            print(f"\n  --- Cliff-blind example #{rank} (Morgan rank #{rank}) ---")
            print(f"  Mol A (idx {i}): {smiles_list[i]}")
            print(f"  Mol B (idx {j}): {smiles_list[j]}")
            print(f"  |Δy| = {ex.delta_y:.2f}, graph_distance = {ex.graph_distance}")
            print(f"\n  {'Fingerprint':<30} {'Similarity':>12}  {'Metric'}")
            print(f"  {'-'*56}")

            for sid, fp in all_fps.items():
                pair_fp = [CliffPair(
                    i=i, j=j,
                    graph_distance=ex.graph_distance,
                    delta_y=ex.delta_y,
                    mcs_atoms=0,
                )]
                sims = cliff_similarities(fp, pair_fp)
                metric = "Tanimoto" if fp.kind == "binary" else "cosine"
                marker = " *** (ECFP)" if sid == "morgan" else ""
                print(f"  {fp.name:<30} {sims[0]:>12.4f}  {metric}{marker}")


if __name__ == "__main__":
    main()
