"""QA sweep: exercise every data/rendering path the notebook depends on, so we
can be confident no cell throws on any reachable input (errors = disqualifier).

Covers:
  - molecule selector: full gallery + adversarial SMILES (salt, invalid, empty,
    single atom, huge, disconnected, aromatic edge cases)
  - MACCS / Morgan / classical explorers: on_bits, bit_hit, highlight_svg,
    fingerprint_strip_svg, collisions, collision_svg, collision_curve
  - Section 4 cliffs: pair_svgs, fold_change, fingerprint_scores for every
    cliff in every target pair
  - Section 4b poses + interaction fingerprints for every posed cliff
  - Section 5 learned-FP grid: load_grid + RMSE fallback + scatter assembly for
    every pair, every alpha, every cliff highlight
Anything that raises is reported; a clean run prints ALL PASS.
"""
from __future__ import annotations

import traceback

from rdkit import Chem

from fingerprints import classical_explorer as ce
from fingerprints import chemeleon_fp as chf
from fingerprints import cliff_view as cv
from fingerprints import gallery as gal
from fingerprints import learned_fp_view as lfv
from fingerprints import maccs_explorer as mx
from fingerprints import morgan_explorer as me
from fingerprints import pose_view as pv
from fingerprints.data import context_cliffs as ctx

FAILURES: list[str] = []


def check(name, fn):
    try:
        fn()
    except Exception:
        FAILURES.append("%s\n%s" % (name, traceback.format_exc()))


# --- adversarial + gallery SMILES ---------------------------------------
ADVERSARIAL = [
    "",  # empty
    "   ",  # whitespace
    "not_a_smiles",  # garbage
    "C[Pt](N)(N)Cl",  # metal complex
    "[Na+].[Cl-]",  # salt / disconnected
    "C",  # single atom (methane)
    "c1ccccc1",  # benzene
    "O",  # water
    "[H][H]",  # explicit hydrogens
    "C" * 200,  # very long chain
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "invalid)(smiles",  # bad parens
    "C1CC1C2CC2.C3CC3",  # multi-fragment
]
GALLERY_SMILES = [g.smiles for g in gal.GALLERY]
ALL_SMILES = GALLERY_SMILES + ADVERSARIAL


def exercise_mol(smi):
    mol = mx.mol_from_smiles(smi)
    valid = mol is not None
    # MACCS paths
    if valid:
        bits = mx.all_bits()
        mx.highlight_svg(mol, mx.blank_hit(), width=200, height=150)
        for b in bits[:3] + bits[-3:]:
            hit = mx.bit_hit(mol, b)
            mx.highlight_svg(mol, hit, width=200, height=150)
            mx.fingerprint_strip_svg(mol, b, width=300, height=30)
        # Morgan paths
        on = me.on_bits(mol)
        for b in (on[:3] + on[-3:]) if on else []:
            hit = me.bit_hit(mol, b)
            me.highlight_svg(mol, hit, width=200, height=150)
            me.fingerprint_strip_svg(mol, b, width=300, height=30)
        cols = me.find_collisions(mol, n_bits=8)
        for c in cols:
            me.collision_svg(mol, c, width=200, height=150)
        me.collision_curve(mol)
        # Classical toolbox
        for key in ("rdkit_topo", "atom_pair", "top_torsion"):
            kon = ce.on_bits(mol, key)
            for b in (kon[:2]) if kon else []:
                hit = ce.bit_hit(mol, key, b)
                ce.highlight_svg(mol, hit, width=200, height=150)
                ce.fingerprint_strip_svg(mol, key, b, width=300, height=30)
        # CheMeleon learned fingerprint + per-dimension heatmap
        chf.fingerprint(mol)
        dims = chf.most_active_dims(mol, k=5)
        for dm in dims[:3]:
            chf.atom_contributions(mol, dm)
            chf.heatmap_svg(mol, dm, width=200, height=150)


for smi in ALL_SMILES:
    check("molecule path: %r" % smi[:40], lambda s=smi: exercise_mol(s))


# --- Section 4: cliffs ---------------------------------------------------
def exercise_cliffs():
    for tp in ctx.TARGET_PAIRS:
        for i, c in enumerate(tp.cliffs):
            cv.pair_svgs(c, width=200, height=150)
            cv.fold_change(max(c.delta_a, c.delta_b))
            cv.fingerprint_scores(c)


check("section 4 cliffs", exercise_cliffs)


# --- Section 4b: poses + interaction fingerprints ------------------------
def exercise_poses():
    for tp in ctx.TARGET_PAIRS:
        for i, c in enumerate(tp.cliffs):
            if not pv.has_poses(tp.key, i):
                continue
            poses = pv.load_all(tp.key, i)
            for p in poses.values():
                pv.load_interactions(p)
                pv.pocket_residues(p.cif_text)
            keys = [
                "mol1_%s" % tp.target_a, "mol2_%s" % tp.target_a,
                "mol1_%s" % tp.target_b, "mol2_%s" % tp.target_b,
            ]
            pv.aligned_interaction_fingerprints(poses, keys)


check("section 4b poses + interaction FPs", exercise_poses)


# --- Section 5: learned-FP grid ------------------------------------------
def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def rmse_of(row, key):
    import math

    m = row.get("rmse_%s_mean" % key)
    if m is not None:
        return m
    sc = (row.get("scatter") or {}).get(key)
    if not sc:
        return float("nan")
    pairs = [
        (a, p)
        for a, p in zip(sc["actual"], sc["pred"])
        if a is not None and p is not None and a == a and p == p
    ]
    if len(pairs) < 5:
        return float("nan")
    return math.sqrt(sum((a - p) ** 2 for a, p in pairs) / len(pairs))


def exercise_learned():
    for pair in lfv.available_pairs():
        g = lfv.load_grid(pair)
        tp = ctx.by_key().get(pair)
        for row in g["results"]:
            rmse_of(row, "a")
            rmse_of(row, "b")
            sc = row.get("scatter") or {}
            if "smiles" not in sc:
                continue
            # highlight every cliff pick
            for ci, c in enumerate((tp.cliffs if tp else [])):
                hi = {}
                for s, lab in ((c.smiles_1, "molecule 1"), (c.smiles_2, "molecule 2")):
                    cs = canon(s)
                    if cs:
                        hi[cs] = lab
                fg = 0
                for i, smi in enumerate(sc["smiles"]):
                    cs = canon(smi)
                    if cs in hi:
                        fg += 1
                # not asserting fg>0 (train/test split may vary), just no throw


check("section 5 learned-FP grid + highlights", exercise_learned)


# --- report --------------------------------------------------------------
print("=" * 60)
if FAILURES:
    print("QA FAILURES: %d" % len(FAILURES))
    for f in FAILURES:
        print("-" * 60)
        print(f)
else:
    print("ALL PASS — no path raised")
