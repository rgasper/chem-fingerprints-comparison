"""Offline: fold context-cliff ligands into both receptors with the Boltz API.

The activity-cliff section's payoff is a 3D "why": the same small change is a
potency cliff on one receptor and flat on a related one. To *show* that, we
co-fold each of a cliff pair's two ligands into *both* receptors (4 predicted
complexes) and look at where the changed atoms land in each pocket.

This calls the hosted Boltz-2 API (needs BOLTZ_API_KEY) and is slow + costs
credits, so it runs **offline**, once. It caches each predicted complex (mmCIF)
plus a small metadata JSON under ``data/boltz_poses/`` for the notebook to load
instantly. Predicted poses are hypotheses, not measurements - the notebook
labels them as such.

Usage:
    export BOLTZ_API_KEY=...   # or put it in .env
    uv run python scripts/boltz_fold_cliffs.py            # default pair
    uv run python scripts/boltz_fold_cliffs.py --pair mu_vs_kappa --index 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from loguru import logger

from fingerprints.data.context_cliffs import by_key


# UniProt accessions for the receptors used in the curated target pairs.
UNIPROT = {
    "Dopamine D3": "P35462",
    "Dopamine D4": "P21917",
    "mu-opioid": "P35372",
    "kappa-opioid": "P41145",
}

OUT_DIR = Path("data/boltz_poses")


def receptor_sequence(target_label: str) -> str:
    up = UNIPROT[target_label]
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{up}.fasta", timeout=60)
    r.raise_for_status()
    return "".join(r.text.splitlines()[1:])


def fold_one(client, seq: str, ligand_smiles: str):
    """Submit one protein+ligand co-fold and block until it finishes."""
    # .run() submits, polls, and returns the completed prediction.
    result = client.predictions.structure_and_binding.run(
        input={
            "entities": [
                {"type": "protein", "chain_ids": ["A"], "value": seq},
                {"type": "ligand_smiles", "chain_ids": ["L"], "value": ligand_smiles},
            ],
            "binding": {
                "type": "ligand_protein_binding",
                "binder_chain_id": "L",
            },
            "num_samples": 1,
        },
        model="boltz-2.1",
    )
    return result


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mu_vs_kappa", help="target-pair key")
    ap.add_argument("--index", type=int, default=2, help="cliff index within the pair")
    ap.add_argument(
        "--force", action="store_true", help="refold even if cached (re-spends credits)"
    )
    args = ap.parse_args()

    try:
        from boltz_api import Boltz
    except ImportError as e:
        raise SystemExit("pip/uv add boltz-api first") from e

    client = Boltz(base_url="https://api.boltz.bio")  # reads BOLTZ_API_KEY

    tp = by_key()[args.pair]
    cliff = tp.cliffs[args.index]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seqs = {
        tp.target_a: receptor_sequence(tp.target_a),
        tp.target_b: receptor_sequence(tp.target_b),
    }

    manifest: list[dict] = []
    for mol_id, smiles in [("mol1", cliff.smiles_1), ("mol2", cliff.smiles_2)]:
        for target, seq in seqs.items():
            tag = f"{args.pair}_{args.index}_{mol_id}_{target.replace(' ', '')}"
            cif_path = OUT_DIR / f"{tag}.cif"
            meta_path = OUT_DIR / f"{tag}.json"

            # Idempotent: if this fold is already cached, reuse it and don't
            # spend Boltz credits again. Delete the .cif to force a refold.
            if cif_path.exists() and meta_path.exists() and not args.force:
                logger.info(f"{tag}: already cached, skipping")
                manifest.append(json.loads(meta_path.read_text()))
                continue

            logger.info(f"folding {tag} ({len(seq)} aa + {smiles})")
            result = fold_one(client, seq, smiles)
            if result.status != "succeeded" or result.output is None:
                logger.error(f"{tag}: status={result.status} error={result.error}")
                continue
            best = result.output.best_sample
            _download(best.structure.url, cif_path)
            bm = result.output.binding_metrics
            entry = {
                "tag": tag,
                "pair": args.pair,
                "index": args.index,
                "mol_id": mol_id,
                "smiles": smiles,
                "target": target,
                "cif_file": cif_path.name,
                "binding_confidence": getattr(bm, "binding_confidence", None),
            }
            # Cache per-fold metadata too, so a partial re-run can resume.
            meta_path.write_text(json.dumps(entry, indent=2))
            manifest.append(entry)
            logger.info(
                f"  saved {cif_path.name} "
                f"(binding_confidence={getattr(bm, 'binding_confidence', None)})"
            )

    (OUT_DIR / f"{args.pair}_{args.index}_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    logger.info(f"wrote manifest with {len(manifest)} poses to {OUT_DIR}")


if __name__ == "__main__":
    main()
