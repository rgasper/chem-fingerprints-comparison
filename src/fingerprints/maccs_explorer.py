"""Helpers for interactively exploring MACCS keys on a single molecule.

MACCS keys are 166 predefined structural patterns (RDKit exposes 167 bits;
index 0 is unused). Each key is a SMARTS pattern plus a count threshold, so a
bit is "on" when the molecule contains at least that many matches. This makes
MACCS the most human-readable fingerprint: every bit maps to a named
substructure you can point at on the molecule.

This module turns that mapping into something a UI can drive: given a molecule
and a bit index, tell me which atoms/bonds lit the bit and render a highlighted
depiction.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MACCSkeys import smartsPatts


# Short human-readable names for the more intuitive keys. RDKit ships only the
# SMARTS, not names; these are curated so the explorer can say what a bit *means*
# in plain language. Missing entries fall back to the raw SMARTS.
BIT_NAMES: dict[int, str] = {
    2: "Group IV/V/VI atom (Si, P, S, ...)",
    3: "actinide / lanthanide",
    19: "seven-membered ring",
    22: "three-membered ring",
    37: "N-heterocycle (N in ring)",
    42: "fluorine",
    46: "N-H",
    49: "8-membered ring atom",
    103: "chlorine",
    104: "nitrogen doubly bonded",
    118: "two adjacent CH2 groups",
    123: "O-C-O (acetal / carboxyl / ether-ester)",
    124: "carbonyl + neighbor pattern",
    134: "halogen",
    136: "C=O (carbonyl)",
    139: "O-H (hydroxyl)",
    154: "C=O (carbonyl, generic)",
    157: "C-O single bond",
    159: "oxygen",
    160: "aliphatic carbon",
    161: "nitrogen",
    162: "aromatic atom",
    163: "six-membered ring",
    164: "oxygen (any)",
    165: "atom in a ring",
}

# Bits with no SMARTS pattern (count-based / hardcoded keys in RDKit).
SPECIAL_BITS: frozenset[int] = frozenset(
    i for i, (smarts, _count) in smartsPatts.items() if smarts == "?"
)


@dataclass(frozen=True)
class BitHit:
    """The result of asking 'does bit N fire on this molecule, and where?'"""

    bit: int
    is_on: bool
    smarts: str | None  # None for special (count-based) keys
    name: str
    match_count: int  # number of distinct substructure matches
    atoms: tuple[int, ...]  # union of atoms across all matches (for highlight)
    bonds: tuple[int, ...]  # bonds fully inside the matched atom set


def bit_name(bit: int) -> str:
    """Human-readable label for a MACCS bit, falling back to its SMARTS."""
    if bit in BIT_NAMES:
        return BIT_NAMES[bit]
    patt = smartsPatts.get(bit)
    if patt is None or patt[0] == "?":
        return f"key {bit} (count-based)"
    return patt[0]


def on_bits(mol: Chem.Mol) -> list[int]:
    """MACCS bit indices that are set for this molecule (1..166)."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return [b for b in fp.GetOnBits() if b != 0]


def bit_hit(mol: Chem.Mol, bit: int) -> BitHit:
    """Resolve a MACCS bit against a molecule: is it on, and which atoms fired?"""
    fp = MACCSkeys.GenMACCSKeys(mol)
    is_on = bool(fp[bit])
    patt = smartsPatts.get(bit)
    smarts = None if (patt is None or patt[0] == "?") else patt[0]

    atoms: set[int] = set()
    match_count = 0
    if smarts is not None:
        query = Chem.MolFromSmarts(smarts)
        if query is not None:
            matches = mol.GetSubstructMatches(query)
            match_count = len(matches)
            for m in matches:
                atoms.update(m)

    bonds: list[int] = []
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms:
            bonds.append(b.GetIdx())

    return BitHit(
        bit=bit,
        is_on=is_on,
        smarts=smarts,
        name=bit_name(bit),
        match_count=match_count,
        atoms=tuple(sorted(atoms)),
        bonds=tuple(bonds),
    )


def highlight_svg(
    mol: Chem.Mol,
    hit: BitHit,
    *,
    width: int = 480,
    height: int = 360,
    color: tuple[float, float, float] = (1.0, 0.55, 0.0),
) -> str:
    """Render mol as SVG with the bit's matched atoms/bonds highlighted."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
    atom_colors = {a: color for a in hit.atoms}
    bond_colors = {b: color for b in hit.bonds}
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


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse SMILES, returning None on failure (for UI validation)."""
    return Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
