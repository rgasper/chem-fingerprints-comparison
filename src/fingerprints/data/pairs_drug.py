"""Drug-sized molecule pairs (~300 Da, ~20 heavy atoms).

Same scaffold/decoration matrix as the small-molecule pairs, but now in the
size range where fingerprints are normally used in medicinal chemistry.

- pair_a: same scaffold (4-aminoquinazoline), small substituent change (Cl -> Br on phenyl)
- pair_b: same scaffold (4-aminoquinazoline), big decoration change (simple aniline -> morpholine + ethyl ether)
- pair_c: different scaffolds (quinazoline vs quinoline), similar decorations (4-Me-anilino)
- pair_d: completely different scaffolds and decorations (celecoxib vs telmisartan-like)

All eight molecules are distinct.
"""

from fingerprints.data.pairs import MoleculePair


PAIRS_DRUG: tuple[MoleculePair, ...] = (
    MoleculePair(
        name="pair_a_drug_same_scaffold_minor",
        description="4-aminoquinazoline; halogen swap on aniline",
        smiles_left="Clc1ccc(Nc2ncnc3ccccc23)cc1",
        label_left="4-Cl-aniline-Q",
        smiles_right="Brc1ccc(Nc2ncnc3ccccc23)cc1",
        label_right="4-Br-aniline-Q",
    ),
    MoleculePair(
        name="pair_b_drug_same_scaffold_major",
        description="4-aminoquinazoline; minimal vs heavily decorated",
        smiles_left="c1ccc2ncnc(Nc3ccccc3)c2c1",
        label_left="aniline-Q",
        smiles_right="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        label_right="gefitinib",
    ),
    MoleculePair(
        name="pair_c_drug_diff_scaffold_same_decor",
        description="naphthalene vs biphenyl (fused vs linked) with shared OMe + Cl + N-methyl-amide decorations",
        smiles_left="COc1ccc2c(Cl)ccc(C(=O)NC)c2c1",
        label_left="decorated naphthalene",
        smiles_right="COc1ccc(-c2ccc(Cl)cc2)cc1C(=O)NC",
        label_right="decorated biphenyl",
    ),
    MoleculePair(
        name="pair_d_drug_all_different",
        description="celecoxib (pyrazole-sulfonamide COX-2) vs telmisartan-like (benzimidazole ARB)",
        smiles_left="Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        label_left="celecoxib",
        smiles_right="CCCCc1nc2cc(C(=O)O)ccc2n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
        label_right="telmisartan-like",
    ),
)


def all_smiles_drug() -> list[str]:
    out: list[str] = []
    for p in PAIRS_DRUG:
        out.append(p.smiles_left)
        out.append(p.smiles_right)
    return out


def all_labels_drug() -> list[str]:
    out: list[str] = []
    for p in PAIRS_DRUG:
        out.append(p.label_left)
        out.append(p.label_right)
    return out
