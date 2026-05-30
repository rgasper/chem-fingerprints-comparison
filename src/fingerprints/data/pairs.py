"""Hand-picked molecule pairs for fingerprint comparisons.

Four pairs designed to vary scaffold and decoration independently:
- pair_a: same scaffold, minor decoration change
- pair_b: same scaffold, major decoration change
- pair_c: different scaffold, similar decorations
- pair_d: completely different scaffold and decorations
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoleculePair:
    name: str
    description: str
    smiles_left: str
    label_left: str
    smiles_right: str
    label_right: str


PAIRS: tuple[MoleculePair, ...] = (
    MoleculePair(
        name="pair_a_same_scaffold_minor",
        description="Same scaffold (benzene), minor decoration change (methyl -> ethyl)",
        smiles_left="Cc1ccc(O)cc1",
        label_left="p-cresol",
        smiles_right="CCc1ccc(O)cc1",
        label_right="4-ethylphenol",
    ),
    MoleculePair(
        name="pair_b_same_scaffold_major",
        description="Same scaffold (indole), major decoration change",
        smiles_left="Cc1[nH]c2ccccc2c1",
        label_left="3-methylindole",
        smiles_right="O=C(O)CCc1[nH]c2ccc(Cl)cc2c1C(=O)N",
        label_right="heavily-decorated indole",
    ),
    MoleculePair(
        name="pair_c_diff_scaffold_same_decor",
        description="Different scaffolds, similar decorations (-OH, -CH3, -Cl)",
        smiles_left="Cc1cc(Cl)cc(O)c1",
        label_left="chloro-methyl-phenol",
        smiles_right="Cc1cc(Cl)nc(O)c1",
        label_right="chloro-methyl-pyridinol",
    ),
    MoleculePair(
        name="pair_d_all_different",
        description="Completely different scaffold and decorations",
        smiles_left="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        label_left="caffeine",
        smiles_right="OC(=O)CCCCCCC/C=C\\CCCCCCCC",
        label_right="oleic acid",
    ),
)


def all_smiles() -> list[str]:
    out: list[str] = []
    for p in PAIRS:
        out.append(p.smiles_left)
        out.append(p.smiles_right)
    return out


def all_labels() -> list[str]:
    out: list[str] = []
    for p in PAIRS:
        out.append(p.label_left)
        out.append(p.label_right)
    return out
