"""Bemis-Murcko scaffold extraction and scaffold-purity-at-k metrics.

A molecule's *scaffold* (Bemis & Murcko 1996) is the union of its rings and the
atoms on the shortest paths between them, with side chains stripped. Two
molecules share a scaffold if their canonical scaffold SMILES match.

Acyclic molecules have an empty Murcko scaffold; we put them all in a
synthetic "_ACYCLIC_" bucket so they only match each other.

The headline metric, *scaffold purity at k*, is leave-one-out: for each
molecule, how many of its k nearest neighbors in fingerprint space share its
scaffold? Averaged across the dataset, this measures the fingerprint's bias
toward grouping molecules by scaffold.

- Purity ~ baseline (= same-scaffold rate in the sample) means the
  fingerprint is scaffold-blind in its nearest-neighbor ordering.
- Purity >> baseline means the fingerprint preferentially retrieves same-
  scaffold neighbors. Pair this with property kNN R\u00b2 to read it as a
  Pareto: high purity + high R\u00b2 = scaffold memorization; low purity + high
  R\u00b2 = genuine scaffold hopping (the fingerprint spans chemotypes while
  staying property-aware).

Example:
    >>> ids = scaffold_ids(["c1ccccc1O", "Cc1ccc(O)cc1"])
    >>> ids[0] == ids[1]  # both phenol scaffold = c1ccccc1
    True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.neighbors import NearestNeighbors
from typeguard import typechecked

from fingerprints.fingerprint_methods.base import (
    FingerprintResult,
    default_metric_for,
)


ACYCLIC_BUCKET = "_ACYCLIC_"


@dataclass(frozen=True)
class ScaffoldPurityResult:
    """Scaffold-purity-at-k metrics for one fingerprint.

    Attributes:
        name: fingerprint display name
        ks: list of k values evaluated
        purity_at_k: parallel array of mean scaffold purity at each k
        baseline: probability that two random molecules share a scaffold,
            i.e. sum_s p_s^2 where p_s is the scaffold's frequency. This is
            the value purity would take for a scaffold-blind fingerprint.
        n_unique_scaffolds: number of distinct scaffolds in the dataset
            (including the synthetic acyclic bucket if present)
    """

    name: str
    ks: list[int]
    purity_at_k: np.ndarray
    baseline: float
    n_unique_scaffolds: int


@typechecked
def murcko_scaffold_smiles(mol: Mol) -> str:
    """Canonical SMILES of the molecule's Bemis-Murcko scaffold.

    Returns ACYCLIC_BUCKET for fully acyclic molecules, or "" if RDKit
    refuses to produce a scaffold (e.g. bad bond stereo).
    """
    try:
        s = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except (RuntimeError, ValueError) as e:
        logger.debug(f"scaffold extraction failed: {e}")
        return ""
    if not s:
        return ACYCLIC_BUCKET
    return s


@typechecked
def scaffold_ids(smiles_or_mols: list[str] | list[Mol]) -> list[str]:
    """Return the scaffold SMILES (or ACYCLIC_BUCKET) for each input.

    Accepts SMILES strings or RDKit Mols. Invalid SMILES become an empty
    string id (which will not match any other entry).
    """
    out: list[str] = []
    for x in smiles_or_mols:
        if isinstance(x, str):
            mol = MolFromSmiles(x)
        else:
            mol = x
        if mol is None:
            out.append("")
            continue
        out.append(murcko_scaffold_smiles(mol))
    return out


@typechecked
def random_match_baseline(scaffolds: list[str]) -> float:
    """Probability that two uniformly random molecules share a scaffold.

    Computed as sum over scaffolds s of (count_s / n)^2, the standard
    "Simpson concentration" of the scaffold distribution. Empty-string
    scaffolds (parse failures) are excluded.
    """
    cleaned = [s for s in scaffolds if s != ""]
    n = len(cleaned)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for s in cleaned:
        counts[s] = counts.get(s, 0) + 1
    return float(sum((c / n) ** 2 for c in counts.values()))


def _prepare_array(fp: FingerprintResult) -> tuple[np.ndarray, str]:
    """Cast for sklearn NearestNeighbors with the metric matching fp.kind."""
    metric = default_metric_for(fp.kind)
    if fp.kind == "binary":
        arr = fp.array.astype(bool)
    else:
        arr = fp.array.astype(np.float32, copy=False)
    return arr, metric


@typechecked
def scaffold_purity_at_k(
    fp: FingerprintResult,
    scaffolds: list[str],
    ks: list[int],
) -> np.ndarray:
    """Mean fraction of k nearest neighbors that share each query's scaffold.

    Leave-one-out: nearest-neighbor index 0 is the molecule itself and is
    dropped before evaluation.

    Args:
        fp: fingerprint result over n molecules
        scaffolds: parallel list of scaffold ids of length n; entries that
            are the empty string are skipped (treated as having no
            scaffold info, so neither query nor neighbor counts).
        ks: list of k values to evaluate

    Returns:
        purity: array of shape (len(ks),) with mean per-molecule purity at
        each k, averaged over molecules with non-empty scaffolds.
    """
    arr, metric = _prepare_array(fp)
    n = arr.shape[0]
    if len(scaffolds) != n:
        raise ValueError(
            f"scaffolds has {len(scaffolds)} entries, fp has {n}"
        )

    max_k = max(ks)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric=metric)
    nn.fit(arr)
    _, idx = nn.kneighbors(arr)
    idx = idx[:, 1:]  # drop self

    # Mask of molecules with a usable scaffold id (non-empty)
    scaffolds_arr = np.array(scaffolds)
    valid = scaffolds_arr != ""
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.zeros(len(ks), dtype=np.float64)

    out = np.zeros(len(ks), dtype=np.float64)
    for ki, k in enumerate(ks):
        neigh = idx[:, :k]
        # For each row, count fraction of neighbors with same scaffold id
        same = scaffolds_arr[neigh] == scaffolds_arr[:, None]
        per_mol = same.mean(axis=1)
        # Average only over valid query molecules
        out[ki] = float(per_mol[valid].mean())
    return out


@typechecked
def scaffold_purity_for(
    fp: FingerprintResult,
    scaffolds: list[str],
    ks: list[int],
) -> ScaffoldPurityResult:
    logger.info(f"scaffold purity: {fp.name} (n={len(scaffolds)})")
    purity = scaffold_purity_at_k(fp, scaffolds, ks)
    baseline = random_match_baseline(scaffolds)
    n_unique = len({s for s in scaffolds if s != ""})
    return ScaffoldPurityResult(
        name=fp.name,
        ks=list(ks),
        purity_at_k=purity,
        baseline=baseline,
        n_unique_scaffolds=n_unique,
    )


@typechecked
def scaffold_purity_for_all(
    fps: dict[str, FingerprintResult],
    scaffolds: list[str],
    ks: list[int],
) -> dict[str, ScaffoldPurityResult]:
    return {sid: scaffold_purity_for(fp, scaffolds, ks) for sid, fp in fps.items()}
