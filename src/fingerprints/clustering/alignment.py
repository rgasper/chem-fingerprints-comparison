"""Statistics that quantify how well a fingerprint's geometry aligns with
an external molecular property (e.g. logS, logD, BBB).

Two complementary lenses:

- **Local alignment** via leave-one-out kNN: each molecule's property is
  predicted from its k nearest neighbors in fingerprint space (excluding
  itself). R\u00b2 (regression) or ROC-AUC (binary classification) measures
  how well the fingerprint's local neighborhoods agree with the property.

- **Global alignment** via Spearman rank correlation: for many random
  pairs of molecules, compute (fingerprint distance, |property_a -
  property_b|) and rank-correlate. High correlation = "molecules far
  apart in fingerprint space have different property values."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from typeguard import typechecked

from fingerprints.fingerprint_methods.base import (
    FingerprintResult,
    default_metric_for,
)


@dataclass(frozen=True)
class AlignmentResult:
    """Per-(fingerprint, property) alignment metrics.

    Attributes:
        name: fingerprint display name
        task_type: 'regression' | 'classification'
        ks: list of k values for the kNN sweep
        knn_scores: parallel array of leave-one-out kNN scores (R\u00b2 for
            regression, ROC-AUC for classification) at each k
        spearman_rho: global rank correlation between fingerprint distance
            and absolute property difference (regression) or 0/1 indicator
            difference (classification). Always in [-1, 1]; positive =
            distant pairs differ more in property.
        spearman_p: two-sided p-value for the Spearman correlation
        n_pairs: number of pairs sampled for Spearman computation
    """

    name: str
    task_type: str
    ks: list[int]
    knn_scores: np.ndarray
    spearman_rho: float
    spearman_p: float
    n_pairs: int


def _prepare_array(fp: FingerprintResult) -> tuple[np.ndarray, str]:
    """Cast the fingerprint to a form suitable for scikit-learn's NearestNeighbors,
    using the metric matching the fingerprint kind."""
    metric = default_metric_for(fp.kind)
    if fp.kind == "binary":
        # sklearn's jaccard expects boolean
        arr = fp.array.astype(bool)
    else:
        arr = fp.array.astype(np.float32, copy=False)
    return arr, metric


@typechecked
def loo_knn_scores(
    fp: FingerprintResult,
    y: np.ndarray,
    task_type: str,
    ks: list[int],
) -> np.ndarray:
    """Leave-one-out kNN scores at each k.

    For regression (R\u00b2): predict each molecule's y from the mean of its k
    nearest neighbors' y values (excluding itself).
    For classification (ROC-AUC): predict probability = fraction of positive
    neighbors among the k nearest, then ROC-AUC on these probabilities.

    Args:
        fp: fingerprint result
        y: property values, shape (n_mols,). Float for regression, 0/1 for
            classification.
        task_type: 'regression' | 'classification'
        ks: list of k values to evaluate

    Returns:
        scores: array of shape (len(ks),) with R\u00b2 (regression) or ROC-AUC
        (classification) at each k.
    """
    arr, metric = _prepare_array(fp)
    n = arr.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows, fp has {n}")

    max_k = max(ks)
    # Fit kNN once with max_k+1 neighbors (the +1 is to drop self)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric=metric)
    nn.fit(arr)
    _, idx = nn.kneighbors(arr)
    # idx[:, 0] is self for every row (distance 0); drop it
    idx = idx[:, 1:]  # (n, max_k)

    out = np.zeros(len(ks), dtype=np.float64)
    for ki, k in enumerate(ks):
        neigh = idx[:, :k]  # (n, k)
        neighbor_y = y[neigh]  # (n, k)
        if task_type == "regression":
            y_pred = neighbor_y.mean(axis=1)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            out[ki] = r2
        elif task_type == "classification":
            # Fraction of neighbors that are positive class
            prob = neighbor_y.astype(float).mean(axis=1)
            try:
                out[ki] = roc_auc_score(y, prob)
            except ValueError as e:
                logger.warning(f"ROC-AUC failed for {fp.name} k={k}: {e}")
                out[ki] = 0.5
        else:
            raise ValueError(f"unknown task_type {task_type!r}")
    return out


@typechecked
def distance_property_spearman(
    fp: FingerprintResult,
    y: np.ndarray,
    n_pairs: int = 50_000,
    seed: int = 0,
) -> tuple[float, float, int]:
    """Spearman rank correlation between fingerprint pairwise distance and
    absolute property difference.

    Sampled (not all O(N\u00b2) pairs) for tractability on n>~5000.

    Returns (rho, p_value, n_pairs_used).
    """
    arr, metric = _prepare_array(fp)
    n = arr.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows, fp has {n}")

    rng = np.random.default_rng(seed)
    # Sample pair indices (i, j) with i != j
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    same = i == j
    # resample any self-pairs
    while same.any():
        j[same] = rng.integers(0, n, size=int(same.sum()))
        same = i == j

    # Compute pairwise distances for the (i, j) pairs only.
    if fp.kind == "binary":
        # Vectorized jaccard distance on bit vectors:
        #   J = 1 - |A AND B| / |A OR B|
        a = arr[i]  # (n_pairs, n_features) bool
        b = arr[j]
        intersection = np.sum(a & b, axis=1)
        union = np.sum(a | b, axis=1)
        # avoid division by zero (both vectors all-False)
        same_empty = union == 0
        dists = np.where(
            same_empty, 0.0, 1.0 - intersection / np.maximum(union, 1)
        )
    else:
        # paired_distances supports cosine for continuous embeddings
        from sklearn.metrics.pairwise import paired_distances

        dists = paired_distances(arr[i], arr[j], metric=metric)
    diffs = np.abs(y[i] - y[j])
    rho, p = spearmanr(dists, diffs)
    return float(rho), float(p), int(n_pairs)


@typechecked
def alignment_for(
    fp: FingerprintResult,
    y: np.ndarray,
    task_type: str,
    ks: list[int],
    n_pairs: int = 50_000,
    seed: int = 0,
) -> AlignmentResult:
    """Compute kNN sweep + Spearman \u03c1 for one (fingerprint, property) combo."""
    logger.info(f"alignment: {fp.name} (n={len(y)}, task={task_type})")
    knn = loo_knn_scores(fp, y, task_type, ks)
    rho, p, n = distance_property_spearman(fp, y, n_pairs=n_pairs, seed=seed)
    return AlignmentResult(
        name=fp.name,
        task_type=task_type,
        ks=list(ks),
        knn_scores=knn,
        spearman_rho=rho,
        spearman_p=p,
        n_pairs=n,
    )


@typechecked
def alignment_for_all(
    fps: dict[str, FingerprintResult],
    y: np.ndarray,
    task_type: str,
    ks: list[int],
    n_pairs: int = 50_000,
    seed: int = 0,
) -> dict[str, AlignmentResult]:
    """Compute alignment metrics for every fingerprint in the dict."""
    out: dict[str, AlignmentResult] = {}
    for sid, fp in fps.items():
        out[sid] = alignment_for(fp, y, task_type, ks, n_pairs=n_pairs, seed=seed)
    return out
