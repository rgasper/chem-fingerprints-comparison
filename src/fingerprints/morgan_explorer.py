"""Interactively explore Morgan (ECFP-like) bits on a single molecule.

Morgan fingerprints are the workhorse of modern cheminformatics and the
conceptual opposite of MACCS. Instead of a fixed checklist of named
substructures, Morgan asks, for every atom: "what does the circular
neighborhood out to radius R look like?" It hashes each such atom environment
into a bit. So a Morgan bit has no human-given name — but RDKit can tell us
exactly which atom(s) and radius produced it, which is all we need to highlight
the environment on the molecule.

This module mirrors ``maccs_explorer`` so the notebook can drive both with the
same interaction (scrub bits -> highlight substructure), letting the reader feel
the difference between an expert checklist and a learned/hashed representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D


DEFAULT_RADIUS = 2
DEFAULT_BITS = 2048


@dataclass(frozen=True)
class MorganBitHit:
    """Which atoms/bonds set a Morgan bit, and at what radius."""

    bit: int
    n_instances: int  # how many atom centers produced this bit
    radii: tuple[int, ...]  # distinct radii across those centers
    centers: tuple[int, ...]  # the center atom indices
    atoms: tuple[int, ...]  # union of environment atoms (for highlight)
    bonds: tuple[int, ...]  # union of environment bonds


def _bit_info(mol: Chem.Mol, radius: int, n_bits: int):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=ao)
    return ao.GetBitInfoMap()


def _env_atoms_bonds(mol: Chem.Mol, center: int, radius: int) -> tuple[set[int], set[int]]:
    """Atoms and bonds of the circular environment around `center` at `radius`."""
    if radius == 0:
        return {center}, set()
    env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    atom_map: dict[int, int] = {}
    Chem.PathToSubmol(mol, env_bonds, atomMap=atom_map)
    atoms = set(atom_map.keys()) | {center}
    return atoms, set(env_bonds)


def on_bits(mol: Chem.Mol, radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_BITS) -> list[int]:
    """Sorted Morgan bit indices that are set for this molecule."""
    return sorted(_bit_info(mol, radius, n_bits).keys())


def bit_hit(
    mol: Chem.Mol,
    bit: int,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_BITS,
) -> MorganBitHit:
    """Resolve a Morgan bit: which atom environment(s) produced it, and where."""
    info = _bit_info(mol, radius, n_bits)
    instances = info.get(bit, ())
    atoms: set[int] = set()
    bonds: set[int] = set()
    centers: list[int] = []
    radii: set[int] = set()
    for center, r in instances:
        centers.append(center)
        radii.add(r)
        a, b = _env_atoms_bonds(mol, center, r)
        atoms |= a
        bonds |= b
    return MorganBitHit(
        bit=bit,
        n_instances=len(instances),
        radii=tuple(sorted(radii)),
        centers=tuple(centers),
        atoms=tuple(sorted(atoms)),
        bonds=tuple(sorted(bonds)),
    )


def highlight_svg(
    mol: Chem.Mol,
    hit: MorganBitHit,
    *,
    width: int = 460,
    height: int = 340,
    env_color: tuple[float, float, float] = (0.20, 0.55, 0.95),
    center_color: tuple[float, float, float] = (0.95, 0.35, 0.25),
) -> str:
    """Render mol as SVG with a Morgan bit's environment highlighted.

    The environment atoms are shaded blue; the center atom(s) that define the
    bit are shaded a distinct red so the "grows outward from here" idea is
    visible.
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    atom_colors = {a: env_color for a in hit.atoms}
    for c in hit.centers:
        atom_colors[c] = center_color
    bond_colors = {b: env_color for b in hit.bonds}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(hit.atoms),
        highlightBonds=list(hit.bonds),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
