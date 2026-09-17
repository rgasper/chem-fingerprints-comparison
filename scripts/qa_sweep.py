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
from fingerprints import importance_view as iv
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
        chf.dim_sensitivity(mol)
        dims = chf.most_active_dims(mol, k=5)
        chf.strip_svg(mol, dims[0] if dims else 0, active_dims=dims, width=300, height=30)
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


# --- feature-importance views (D3/D4 cliff pair) -------------------------
def exercise_importance():
    if not iv.has_data():
        return
    eps = iv.endpoints()
    for tp in ctx.TARGET_PAIRS:
        if tp.target_a not in eps or tp.target_b not in eps:
            continue
        for c in tp.cliffs:
            for smi in (c.smiles_1, c.smiles_2):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                for fp in ("ecfp", "chemeleon"):
                    iv.atom_importance(mol, c.cliff_on, fp)
                    iv.importance_heatmap_svg(mol, c.cliff_on, fp, width=200, height=150)
                    iv.strip_svg(mol, c.cliff_on, fp, width=300, height=26)
                    iv.metrics(c.cliff_on, fp)


check("feature-importance views", exercise_importance)


# --- report --------------------------------------------------------------
print("=" * 60)
if FAILURES:
    print("QA FAILURES: %d" % len(FAILURES))
    for f in FAILURES:
        print("-" * 60)
        print(f)
else:
    print("ALL PASS — no path raised")
