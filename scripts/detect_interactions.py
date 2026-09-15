"""Offline: detect protein-ligand interactions in the cached Boltz poses.

Runs [PLIP](https://github.com/pharmai/plip) on each predicted complex to detect
the interactions that hold the ligand in the pocket - salt bridges, hydrogen
bonds, pi-stacking, pi-cation, and hydrophobic contacts - and caches them as a
small JSON per pose (endpoint coordinates for each interaction, so the notebook
can draw them). PLIP protonates the structure and applies published geometric
rules, so "detected by PLIP" is a citable analysis rather than our own guess.

This is slow (openbabel conversion + PLIP, ~15s/pose) and needs extra deps, so
it runs offline once. The notebook only reads the cached JSON.

Note: these are interactions in *predicted* complexes - hypotheses about the
binding mode, not experimental fact. The notebook labels them accordingly.

Usage:
    uv run python scripts/detect_interactions.py         # all cached poses
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

POSE_DIR = Path("data/boltz_poses")


def _cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    from openbabel import openbabel as ob

    conv = ob.OBConversion()
    conv.SetInAndOutFormats("cif", "pdb")
    mol = ob.OBMol()
    conv.ReadFile(mol, str(cif_path))
    conv.WriteFile(mol, str(pdb_path))


def _xyz(coords) -> list[float]:
    return [float(c) for c in coords]


def detect(pdb_path: Path) -> list[dict]:
    """Return a list of interaction records with 3D endpoints, via PLIP."""
    from plip.structure.preparation import PDBComplex

    complex_ = PDBComplex()
    complex_.load_pdb(str(pdb_path))
    for lig in complex_.ligands:
        complex_.characterize_complex(lig)

    out: list[dict] = []
    for site in complex_.interaction_sets.values():
        for hb in list(site.hbonds_ldon) + list(site.hbonds_pdon):
            out.append(
                {
                    "type": "hbond",
                    "residue": f"{hb.restype}{hb.resnr}",
                    "distance": round(float(hb.distance_ad), 2),
                    "lig_xyz": _xyz(hb.a.coords if hb.protisdon else hb.d.coords),
                    "prot_xyz": _xyz(hb.d.coords if hb.protisdon else hb.a.coords),
                }
            )
        for sb in list(site.saltbridge_lneg) + list(site.saltbridge_pneg):
            lig_center = sb.negative.center if sb.protispos else sb.positive.center
            prot_center = sb.positive.center if sb.protispos else sb.negative.center
            out.append(
                {
                    "type": "saltbridge",
                    "residue": f"{sb.restype}{sb.resnr}",
                    "distance": round(float(sb.distance), 2),
                    "lig_xyz": _xyz(lig_center),
                    "prot_xyz": _xyz(prot_center),
                }
            )
        for ps in site.pistacking:
            out.append(
                {
                    "type": "pistack",
                    "residue": f"{ps.restype}{ps.resnr}",
                    "distance": round(float(ps.distance), 2),
                    "lig_xyz": _xyz(ps.ligandring.center),
                    "prot_xyz": _xyz(ps.proteinring.center),
                }
            )
        for pc in list(site.pication_laro) + list(site.pication_paro):
            out.append(
                {
                    "type": "pication",
                    "residue": f"{pc.restype}{pc.resnr}",
                    "distance": round(float(pc.distance), 2),
                    "lig_xyz": _xyz(pc.ring.center if pc.protcharged else pc.charge.center),
                    "prot_xyz": _xyz(pc.charge.center if pc.protcharged else pc.ring.center),
                }
            )
        for hc in site.hydrophobic_contacts:
            out.append(
                {
                    "type": "hydrophobic",
                    "residue": f"{hc.restype}{hc.resnr}",
                    "distance": round(float(hc.distance), 2),
                    "lig_xyz": _xyz(hc.ligatom.coords),
                    "prot_xyz": _xyz(hc.bsatom.coords),
                }
            )
    return out


def main() -> None:
    cifs = sorted(POSE_DIR.glob("*.cif"))
    logger.info(f"detecting interactions for {len(cifs)} poses")
    for cif in cifs:
        out_path = cif.with_suffix(".interactions.json")
        pdb_path = cif.with_suffix(".tmp.pdb")  # scratch, inside the repo
        try:
            _cif_to_pdb(cif, pdb_path)
            records = detect(pdb_path)
        finally:
            pdb_path.unlink(missing_ok=True)
        out_path.write_text(json.dumps(records, indent=2))
        counts: dict[str, int] = {}
        for r in records:
            counts[r["type"]] = counts.get(r["type"], 0) + 1
        logger.info(f"  {cif.stem}: {counts}")


if __name__ == "__main__":
    main()
