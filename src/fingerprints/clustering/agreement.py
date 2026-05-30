"""Pairwise agreement between fingerprints via the RV coefficient.

The RV coefficient is a multivariate generalization of squared correlation
between two configurations of the same set of objects (Robert & Escoufier,
1976). Given two fingerprints A, B over the same n molecules:

    RV(A, B) = trace(S_A S_B) / sqrt(trace(S_A^2) * trace(S_B^2))

where S_X = X X^T is the n-by-n Gram matrix of (centered) features. RV is in
[0, 1]; 1 means the two configurations are identical up to rotation/scale,
0 means they capture orthogonal structure.

We center each fingerprint's column-wise mean before forming the Gram matrix,
which removes a trivial "all rows are similar in the same constant way"
component. Binary fingerprints are cast to float for centering.

For tractability we never instantiate the (n_features x n_features) covariance
matrices - all computation goes through n-by-n Gram matrices, which we form
once per fingerprint.

Example:
    >>> fps = {"morgan": morgan_fp, "chemeleon": chemeleon_fp}
    >>> rv = rv_matrix(fps)
    >>> rv["morgan"]["chemeleon"]  # doctest: +SKIP
    0.42
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from typeguard import typechecked

from fingerprints.fingerprint_methods.base import FingerprintResult


@dataclass(frozen=True)
class AgreementResult:
    """Pairwise RV-coefficient matrix across a set of fingerprints.

    Attributes:
        short_ids: ordered list of short fingerprint ids (column / row order)
        display_names: parallel list of human-readable names
        rv: (k, k) symmetric matrix of RV coefficients in [0, 1]
        n_molecules: number of molecules each fingerprint was computed on
    """

    short_ids: list[str]
    display_names: list[str]
    rv: np.ndarray
    n_molecules: int


def _centered_float(fp: FingerprintResult) -> np.ndarray:
    """Cast to float32 and column-center.

    Centering removes the constant offset that bit fingerprints inevitably
    have (most bits are 0 for any individual molecule); without it RV is
    inflated for sparse binary fingerprints.
    """
    arr = fp.array.astype(np.float32, copy=False)
    return arr - arr.mean(axis=0, keepdims=True)


def _gram(arr: np.ndarray) -> np.ndarray:
    """n x n Gram matrix S = X X^T."""
    return arr @ arr.T


@typechecked
def rv_coefficient(g_a: np.ndarray, g_b: np.ndarray) -> float:
    """RV coefficient between two precomputed Gram matrices.

    Both matrices must be n x n on the same n objects.
    """
    if g_a.shape != g_b.shape or g_a.shape[0] != g_a.shape[1]:
        raise ValueError(
            f"expected two equal-shape square matrices, got {g_a.shape} {g_b.shape}"
        )
    num = float(np.sum(g_a * g_b))  # trace(A B) = sum of elementwise A*B for symmetric
    den = float(np.sqrt(np.sum(g_a * g_a) * np.sum(g_b * g_b)))
    if den == 0.0:
        return 0.0
    return num / den


@typechecked
def rv_matrix(fps: dict[str, FingerprintResult]) -> AgreementResult:
    """Compute the symmetric RV-coefficient matrix across all fingerprints.

    Each fingerprint is centered, then its n x n Gram matrix is formed once
    and reused for every pairing. Memory cost is O(k * n^2) which is the
    dominant term for n in the low thousands and k=8.

    Args:
        fps: dict short_id -> FingerprintResult, all over the same n molecules
            in the same order.

    Returns:
        AgreementResult with row/column order matching fps.keys() insertion order.
    """
    short_ids = list(fps.keys())
    if len(short_ids) < 2:
        raise ValueError(f"need at least 2 fingerprints, got {len(short_ids)}")

    n_set = {fp.array.shape[0] for fp in fps.values()}
    if len(n_set) != 1:
        raise ValueError(f"fingerprints have inconsistent n: {n_set}")
    n_mols = n_set.pop()

    logger.info(
        f"computing Gram matrices for {len(short_ids)} fingerprints on "
        f"{n_mols} molecules"
    )
    grams: dict[str, np.ndarray] = {}
    for sid, fp in fps.items():
        arr = _centered_float(fp)
        grams[sid] = _gram(arr)
        logger.info(f"  {sid}: {fp.array.shape[1]}d -> {n_mols}x{n_mols} Gram")

    k = len(short_ids)
    rv = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            rv[i, j] = rv_coefficient(grams[short_ids[i]], grams[short_ids[j]])
            rv[j, i] = rv[i, j]

    display_names = [fps[sid].name for sid in short_ids]
    return AgreementResult(
        short_ids=short_ids,
        display_names=display_names,
        rv=rv,
        n_molecules=n_mols,
    )
