"""A palette of small, chemically real edits for the fingerprint playground.

Each *edit* takes an RDKit molecule and returns a **new, sanitized, valid**
molecule with one medicinal-chemistry-style change applied - or ``None`` if the
edit doesn't apply to this molecule (no suitable site). Nothing here ever
raises: the playground shows only the edits that produced a valid result, so a
user poking at a molecule can never land the notebook in an error state.

The point of the playground: apply a tiny, legible change (add a fluorine,
methylate a ring, swap an amine for an amide bioisostere) and watch how the
ECFP bits and the CheMeleon dimensions move. Some edits barely perturb the
fingerprint; some flip many bits. That is the whole lesson of the notebook made
tactile - a fingerprint is a *chosen* notion of similarity, and different edits
land very differently under that choice.

Edits are expressed as RDKit reaction SMARTS where possible (robust, and they
enumerate every matching site) plus a few programmatic edits (e.g. delete a
terminal atom) that are awkward to write as reactions. Every edit runs through
``_finalise`` which sanitizes and returns a canonical, de-duplicated molecule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class Edit:
    """One named transformation the user can apply."""

    key: str
    label: str  # short button text
    description: str  # what it does, in medchem terms
    apply: Callable[[Chem.Mol], Chem.Mol | None]


def _finalise(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Sanitize a candidate product; return None if it isn't a valid molecule."""
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
        clean = Chem.MolFromSmiles(smi)
        if clean is None:
            return None
        Chem.SanitizeMol(clean)
        return clean
    except Exception:
        return None


def _run_reaction(smarts: str) -> Callable[[Chem.Mol], Chem.Mol | None]:
    """Build an edit function from a single-reactant reaction SMARTS.

    Returns the first valid, distinct product (RDKit enumerates one per matching
    site; we take the first that sanitizes and differs from the input).
    """

    def _edit(mol: Chem.Mol) -> Chem.Mol | None:
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
        except Exception:
            return None
        if rxn is None:
            return None
        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            return None
        src = Chem.MolToSmiles(mol)
        for prod_tuple in products:
            cand = _finalise(prod_tuple[0])
            if cand is not None and Chem.MolToSmiles(cand) != src:
                return cand
        return None

    return _edit


def _delete_terminal_heavy(mol: Chem.Mol) -> Chem.Mol | None:
    """Remove one terminal (degree-1) heavy atom that isn't in a ring.

    A quick way to *shrink* a molecule and see the fingerprint respond; picks
    the first eligible terminal atom in canonical order.
    """
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1 and not atom.IsInRing():
            rw = Chem.RWMol(mol)
            rw.RemoveAtom(atom.GetIdx())
            return _finalise(rw.GetMol())
    return None


# --- The palette ---------------------------------------------------------
# Reaction SMARTS use atom-map numbers to preserve the matched atom(s) and
# attach/modify a group. They are intentionally simple and legible.

_EDITS: tuple[Edit, ...] = (
    # --- Halogenation: swap an aromatic C-H for a halogen ---
    Edit(
        "aro_f", "Add –F (aromatic)",
        "Fluorinate an aromatic C–H — the classic potency/metabolism tweak.",
        _run_reaction("[cH:1]>>[c:1]F"),
    ),
    Edit(
        "aro_cl", "Add –Cl (aromatic)",
        "Chlorinate an aromatic C–H — bulkier, more lipophilic than fluorine.",
        _run_reaction("[cH:1]>>[c:1]Cl"),
    ),
    Edit(
        "aro_br", "Add –Br (aromatic)",
        "Brominate an aromatic C–H — a heavy halogen, big effect on some FPs.",
        _run_reaction("[cH:1]>>[c:1]Br"),
    ),
    # --- Methylation / alkylation: the 'magic methyl' ---
    Edit(
        "aro_me", "Add –CH₃ (aromatic)",
        "Methylate an aromatic ring — the famous 'magic methyl' effect.",
        _run_reaction("[cH:1]>>[c:1]C"),
    ),
    Edit(
        "n_me", "N-methylate",
        "Add a methyl to an N–H — changes H-bond donor count and shape.",
        _run_reaction("[NX3;H1:1]>>[N:1]C"),
    ),
    # --- Add polar functional groups ---
    Edit(
        "aro_oh", "Add –OH (aromatic)",
        "Hydroxylate an aromatic C–H — adds an H-bond donor (a phenol).",
        _run_reaction("[cH:1]>>[c:1]O"),
    ),
    Edit(
        "aro_ome", "Add –OCH₃ (aromatic)",
        "Add a methoxy to an aromatic ring — H-bond acceptor, common motif.",
        _run_reaction("[cH:1]>>[c:1]OC"),
    ),
    Edit(
        "aro_nh2", "Add –NH₂ (aromatic)",
        "Aminate an aromatic C–H — adds a strong H-bond donor (an aniline).",
        _run_reaction("[cH:1]>>[c:1]N"),
    ),
    Edit(
        "aro_cf3", "Add –CF₃ (aromatic)",
        "Add a trifluoromethyl to an aromatic ring — big lipophilic bump.",
        _run_reaction("[cH:1]>>[c:1]C(F)(F)F"),
    ),
    Edit(
        "aro_cn", "Add –C≡N (aromatic)",
        "Add a nitrile to an aromatic ring — small, strongly polar.",
        _run_reaction("[cH:1]>>[c:1]C#N"),
    ),
    # --- Bioisosteric / functional-group swaps ---
    Edit(
        "cooh_to_tetrazole", "–COOH → tetrazole",
        "Classic carboxylic-acid bioisostere: same acidity, different atoms.",
        _run_reaction("[CX3:1](=O)[OX2H1]>>[c:1]1n[nH]nn1"),
    ),
    Edit(
        "ester_to_amide", "Ester → amide",
        "Swap an ester O for an N–H — the metabolically stabler amide.",
        _run_reaction("[CX3:1](=O)[OX2][#6:3]>>[C:1](=O)[NH][#6:3]"),
    ),
    Edit(
        "oh_to_f", "–OH → –F",
        "Replace a hydroxyl with fluorine — a size-matched H-bond swap.",
        _run_reaction("[CX4:1][OX2H1]>>[C:1]F"),
    ),
    Edit(
        "ketone_to_alcohol", "Ketone → alcohol",
        "Reduce a C=O to C–OH — acceptor becomes donor.",
        _run_reaction("[CX3:1](=O)[#6:2]>>[C:1]([OH])[#6:2]"),
    ),
    Edit(
        "nitro_to_amine", "–NO₂ → –NH₂",
        "Reduce a nitro group to an amine — flips it from acceptor to donor.",
        _run_reaction("[NX3+:1](=O)[O-]>>[NH2:1]"),
    ),
    # --- Ring / heteroatom edits ---
    Edit(
        "benzene_to_pyridine", "Benzene → pyridine",
        "Swap one aromatic C–H for a ring nitrogen — a common aza-scan move.",
        _run_reaction("[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1>>[n:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1"),
    ),
    Edit(
        "aro_ncH_to_n", "Aromatic C → N (aza-swap)",
        "Turn one aromatic C–H into a ring nitrogen anywhere it fits.",
        _run_reaction("[cH:1]>>[n:1]"),
    ),
    # --- Chain edits ---
    Edit(
        "homologate", "Extend a chain (+CH₂)",
        "Insert a methylene into a terminal methyl — homologation.",
        _run_reaction("[CX4H3:1]>>[CH2:1]C"),
    ),
    Edit(
        "delete_terminal", "Delete a terminal atom",
        "Remove one dangling (non-ring) heavy atom — shrink the molecule.",
        _delete_terminal_heavy,
    ),
)

EDITS_BY_KEY: dict[str, Edit] = {e.key: e for e in _EDITS}


def all_edits() -> tuple[Edit, ...]:
    return _EDITS


def applicable_edits(mol: Chem.Mol) -> list[Edit]:
    """Edits that actually produce a valid, distinct product for this molecule.

    Runs each edit once so the UI can offer only the buttons that will do
    something (and never a button that would silently no-op or error).
    """
    if mol is None:
        return []
    out: list[Edit] = []
    for e in _EDITS:
        try:
            if e.apply(mol) is not None:
                out.append(e)
        except Exception:
            continue
    return out


def apply_edit(mol: Chem.Mol, key: str) -> Chem.Mol | None:
    """Apply the named edit, returning a valid product or None."""
    edit = EDITS_BY_KEY.get(key)
    if edit is None or mol is None:
        return None
    try:
        return edit.apply(mol)
    except Exception:
        return None


# --- Diff visualisations -------------------------------------------------
# The playground shows *how the fingerprint responded* to an edit as a picture,
# not a sentence: which ECFP bits flipped, and which CheMeleon dimensions moved
# most. Both renderers take the before/after molecules and return an SVG.

_ECFP_RADIUS = 2
_ECFP_NBITS = 2048
_SHARED = "#ced4da"   # bit on in both (grey)
_ADDED = "#2f9e44"    # bit the edit switched on (green)
_REMOVED = "#e03131"  # bit the edit switched off (red)


def _ecfp_on_bits(mol: Chem.Mol) -> set[int]:
    # Reuse the notebook's canonical Morgan settings so this diff strip is
    # consistent with every other ECFP number in the notebook.
    from fingerprints.morgan_explorer import on_bits

    return set(on_bits(mol, _ECFP_RADIUS, _ECFP_NBITS))


def ecfp_diff_stats(before: Chem.Mol, after: Chem.Mol) -> dict:
    """Tanimoto + counts of shared / added / removed ECFP bits between two mols."""
    b, a = _ecfp_on_bits(before), _ecfp_on_bits(after)
    shared, added, removed = b & a, a - b, b - a
    union = b | a
    tan = (len(shared) / len(union)) if union else 1.0
    return {
        "tanimoto": tan,
        "shared": len(shared),
        "added": len(added),
        "removed": len(removed),
    }


def ecfp_diff_svg(
    before: Chem.Mol, after: Chem.Mol, *, width: int = 920, height: int = 46
) -> str:
    """The ECFP bit vector's response to an edit, as a strip.

    Every ON bit (in either molecule) is a full-height tick coloured by fate:
    grey = unchanged, green = the edit switched it on, red = switched it off.
    Reading it: mostly grey means the fingerprint barely noticed the edit; lots
    of red/green means it saw a big change.
    """
    b, a = _ecfp_on_bits(before), _ecfp_on_bits(after)
    shared, added, removed = b & a, a - b, b - a
    pad = 2
    inner_w = width - 2 * pad
    band_h = height - 2 * pad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="{pad}" y="{pad}" width="{inner_w}" height="{band_h}" '
        f'fill="#f8f9fa" stroke="#dee2e6" stroke-width="0.5" />',
    ]
    tick_w = max(1.0, inner_w / _ECFP_NBITS)
    # Draw shared first, then changed on top so flips are never hidden.
    for bit in sorted(shared):
        x = pad + (bit / _ECFP_NBITS) * inner_w
        parts.append(
            f'<rect x="{x:.2f}" y="{pad + band_h * 0.28:.2f}" '
            f'width="{tick_w:.2f}" height="{band_h * 0.44:.2f}" fill="{_SHARED}" />'
        )
    for bit, color in [(bit, _REMOVED) for bit in removed] + [
        (bit, _ADDED) for bit in added
    ]:
        x = pad + (bit / _ECFP_NBITS) * inner_w
        cw = max(tick_w, 3.0)
        parts.append(
            f'<rect x="{x - cw / 2:.2f}" y="0" width="{cw:.2f}" '
            f'height="{height}" fill="{color}" />'
        )
    parts.append("</svg>")
    return "".join(parts)


def chemeleon_delta_svg(
    before: Chem.Mol,
    after: Chem.Mol,
    *,
    top_k: int = 40,
    width: int = 920,
    height: int = 120,
) -> str | None:
    """The CheMeleon embedding's response as a diverging bar chart.

    CheMeleon dimensions are continuous, so there are no bits to flip - instead
    we show, for the dimensions that moved most, the signed change (after minus
    before): bars up (blue) = the edit pushed that dimension up, down (orange) =
    down. Returns None if the learned weights aren't available.
    """
    try:
        from fingerprints import chemeleon_fp as chf

        vb = chf.fingerprint(before)
        va = chf.fingerprint(after)
    except Exception:
        return None
    import numpy as np

    delta = va - vb
    order = np.argsort(-np.abs(delta))[:top_k]
    if len(order) == 0:
        return None
    mx = float(np.abs(delta[order]).max()) or 1.0
    pad = 6
    inner_w = width - 2 * pad
    mid = height / 2
    bar_w = inner_w / len(order)
    up, down = "#4c6ef5", "#e8820c"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<line x1="{pad}" y1="{mid:.1f}" x2="{width - pad}" y2="{mid:.1f}" '
        f'stroke="#adb5bd" stroke-width="1" />',
    ]
    for i, dim in enumerate(order):
        d = float(delta[dim])
        x = pad + i * bar_w
        h = abs(d) / mx * (mid - pad)
        if d >= 0:
            parts.append(
                f'<rect x="{x:.2f}" y="{mid - h:.2f}" width="{max(bar_w - 1, 1):.2f}" '
                f'height="{h:.2f}" fill="{up}" />'
            )
        else:
            parts.append(
                f'<rect x="{x:.2f}" y="{mid:.2f}" width="{max(bar_w - 1, 1):.2f}" '
                f'height="{h:.2f}" fill="{down}" />'
            )
    parts.append("</svg>")
    return "".join(parts)
