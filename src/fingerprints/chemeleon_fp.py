"""CheMeleon: a *pretrained* learned molecular fingerprint, and per-atom
attribution of individual embedding dimensions.

CheMeleon (Burns et al., 2025) is a message-passing neural network pretrained
on large molecular data; its learned graph embedding is a 2048-dim fingerprint.
Unlike MACCS/Morgan we did NOT design its features - the network learned them.

We load only the message-passing block (`chemeleon_mp.pt`) and use its
per-atom hidden vectors:

    H = mp(batch_mol_graph)          # (n_atoms, 2048)
    fingerprint(mol) = mean(H, dim=0)  # the 2048-dim molecule vector

Because the graph fingerprint is a *mean over atoms*, each atom's contribution
to embedding dimension ``k`` is exactly ``H[i, k] / n_atoms`` - an exact
decomposition (not a saliency approximation). That lets us color a molecule by
"how much each atom drives fingerprint dimension k", the learned analog of the
Morgan bit-scrubber.

Weights are cached under ``.cache/chemeleon``; download once from Zenodo.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

CACHE_DIR = Path(".cache/chemeleon")
WEIGHTS = CACHE_DIR / "chemeleon_mp.pt"
WEIGHTS_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
EMBED_DIM = 2048


def pick_device() -> str:
    """cuda > mps > cpu, whatever this machine has."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_weights() -> Path:
    """Download the CheMeleon MP checkpoint if not already cached."""
    if not WEIGHTS.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    return WEIGHTS


@lru_cache(maxsize=1)
def _load_mp():
    """Load the pretrained BondMessagePassing block onto the best device."""
    import torch
    from chemprop import featurizers, nn

    ckpt = torch.load(ensure_weights(), map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    mp = nn.BondMessagePassing(
        d_v=hp["d_v"],
        d_e=hp["d_e"],
        d_h=hp["d_h"],
        depth=hp["depth"],
        bias=hp["bias"],
        dropout=hp["dropout"],
        activation=hp["activation"],
        undirected=hp["undirected"],
    )
    mp.load_state_dict(ckpt["state_dict"], strict=True)
    mp.eval()
    device = pick_device()
    mp.to(device)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    return mp, feat, device


def atom_hidden(mol: Chem.Mol) -> np.ndarray:
    """Per-atom hidden vectors after message passing: (n_atoms, EMBED_DIM)."""
    import torch
    from chemprop.data import BatchMolGraph

    mp, feat, device = _load_mp()
    bmg = BatchMolGraph([feat(mol)])
    bmg.to(device)
    with torch.no_grad():
        H = mp(bmg).cpu().numpy()
    return H  # (n_atoms, EMBED_DIM)


def fingerprint(mol: Chem.Mol) -> np.ndarray:
    """The 2048-dim CheMeleon fingerprint (mean over atoms)."""
    return atom_hidden(mol).mean(axis=0)


def atom_contributions(mol: Chem.Mol, dim: int) -> np.ndarray:
    """Each atom's exact contribution to embedding dimension ``dim``.

    Since fingerprint[dim] = mean_i H[i, dim], atom i contributes
    H[i, dim] / n_atoms. Returns a (n_atoms,) array.
    """
    H = atom_hidden(mol)
    n = H.shape[0]
    return H[:, dim] / n


@dataclass(frozen=True)
class DimInfo:
    dim: int
    value: float  # fingerprint[dim] (the mean)
    top_atom: int  # atom contributing most (abs)


def most_active_dims(mol: Chem.Mol, k: int = 40) -> list[int]:
    """Embedding dimensions with the largest spread of per-atom contributions -
    i.e. the ones where *which atom you're looking at* matters most, so the
    heatmap is interesting. Returns dim indices, most-varying first."""
    H = atom_hidden(mol)
    spread = H.max(axis=0) - H.min(axis=0)
    order = np.argsort(spread)[::-1]
    # Keep only dims that actually vary across atoms (skip flat/padding dims);
    # fall back to the raw order if a tiny molecule has none.
    varying = [int(d) for d in order if spread[d] > 1e-6]
    picked = varying[:k] if varying else [int(d) for d in order[:k]]
    return picked


def heatmap_svg(mol: Chem.Mol, dim: int, *, width: int = 460, height: int = 340) -> str:
    """Color the molecule by each atom's contribution to dimension ``dim``.

    Green = pushes the dimension up, pink = pushes it down (RDKit similarity-map
    convention). This is an *estimated* read of what the learned dimension keys
    on for this molecule - not a fixed substructure definition like Morgan.
    """
    weights = [float(w) for w in atom_contributions(mol, dim)]
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d.drawOptions().addStereoAnnotation = False
    if mol.GetNumAtoms() < 2:
        # Similarity maps need >=2 atoms; just draw the structure plainly.
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    # If every atom contributes ~equally the map is flat; that's fine.
    SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, draw2d=d)
    d.FinishDrawing()
    return d.GetDrawingText()
