"""Download and cache a 20K-molecule subset of ChEMBL."""

from __future__ import annotations

import gzip
import random
from pathlib import Path

import polars as pl
import requests
from loguru import logger
from rdkit import Chem
from tqdm import tqdm
from typeguard import typechecked

from fingerprints.exceptions import DataLoadError

# ChEMBL provides a SMILES-only file as part of each release: chembl_*_chemreps.txt.gz
# This is much smaller than the SDF/SQLite dump (~150MB gzipped vs many GB).
CHEMBL_VERSION = "35"
CHEMREPS_URL = (
    f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    f"chembl_{CHEMBL_VERSION}/chembl_{CHEMBL_VERSION}_chemreps.txt.gz"
)


@typechecked
def download_chemreps(dest: Path) -> Path:
    """Download chembl_<v>_chemreps.txt.gz to dest if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"chemreps already present at {dest}")
        return dest
    logger.info(f"downloading {CHEMREPS_URL} -> {dest}")
    try:
        with requests.get(CHEMREPS_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc="chembl chemreps"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except requests.RequestException as e:
        raise DataLoadError(f"failed to download {CHEMREPS_URL}: {e}") from e
    return dest


@typechecked
def parse_chemreps(path: Path, max_rows: int | None = None) -> pl.DataFrame:
    """Parse the gzipped chemreps file. Columns:
    chembl_id, canonical_smiles, standard_inchi, standard_inchi_key.

    Returns DataFrame with chembl_id and canonical_smiles columns only.
    """
    rows: list[tuple[str, str]] = []
    with gzip.open(path, "rt") as fh:
        header = fh.readline()
        if "canonical_smiles" not in header:
            raise DataLoadError(f"unexpected header: {header!r}")
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1]))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return pl.DataFrame(
        {"chembl_id": [r[0] for r in rows], "smiles": [r[1] for r in rows]}
    )


@typechecked
def sample_valid_smiles(
    df: pl.DataFrame,
    n: int,
    seed: int = 0,
    max_atoms: int = 60,
    min_atoms: int = 5,
) -> pl.DataFrame:
    """Random-sample n valid drug-like-ish SMILES from df.

    Filters to molecules that parse with RDKit and fall within size bounds.
    Oversamples to account for invalid/oversized molecules.
    """
    rng = random.Random(seed)
    indices = list(range(df.height))
    rng.shuffle(indices)

    keep_chembl: list[str] = []
    keep_smiles: list[str] = []
    keep_canon: list[str] = []

    pbar = tqdm(indices, total=n, desc="filtering smiles")
    for idx in pbar:
        if len(keep_smiles) >= n:
            break
        row = df.row(idx, named=True)
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms < min_atoms or n_atoms > max_atoms:
            continue
        keep_chembl.append(row["chembl_id"])
        keep_smiles.append(smi)
        keep_canon.append(Chem.MolToSmiles(mol))
        pbar.update(0)
        pbar.n = len(keep_smiles)
        pbar.refresh()

    if len(keep_smiles) < n:
        logger.warning(
            f"only collected {len(keep_smiles)}/{n} valid molecules; consider raising sample size"
        )

    return pl.DataFrame(
        {
            "chembl_id": keep_chembl,
            "smiles": keep_smiles,
            "canonical_smiles": keep_canon,
        }
    )
