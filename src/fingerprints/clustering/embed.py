"""Per-fingerprint UMAP + HDBSCAN helpers for the clustering and ADME figures.

Each fingerprint is reduced to 2D independently with a metric matching its
kind (jaccard for binary fingerprints, cosine for continuous embeddings).
HDBSCAN is then run on the original (high-dim) feature space, also with the
matching metric, so cluster identity is not biased by UMAP layout choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np
import umap
from loguru import logger
from typeguard import typechecked

from fingerprints.fingerprint_methods.base import (
    FingerprintResult,
    default_metric_for,
)


@dataclass(frozen=True)
class EmbeddingResult:
    """Output of fitting UMAP (and optionally HDBSCAN) for one fingerprint.

    Attributes:
        name: display name of the fingerprint method
        coords: (n_mols, 2) UMAP coordinates
        cluster_labels: (n_mols,) integer HDBSCAN labels (-1 = noise),
            or None if clustering not run
        n_clusters: number of clusters found (excluding the noise label)
        metric: distance metric used
    """

    name: str
    coords: np.ndarray
    cluster_labels: np.ndarray | None
    n_clusters: int
    metric: str


@typechecked
def umap_embed(
    fp: FingerprintResult,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> np.ndarray:
    """Fit a 2D UMAP for one fingerprint.

    Uses jaccard distance for binary fingerprints, cosine for continuous.
    """
    metric = default_metric_for(fp.kind)
    # umap-learn's jaccard metric expects boolean / 0-1 input. Make sure we
    # pass it that way for binary fingerprints, and cast to float otherwise.
    arr = fp.array
    if fp.kind == "binary":
        arr = arr.astype(bool)
    else:
        arr = arr.astype(np.float32, copy=False)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    logger.info(
        f"UMAP fit: {fp.name} ({arr.shape[0]} mols, {arr.shape[1]}d, metric={metric})"
    )
    coords = reducer.fit_transform(arr)
    return np.asarray(coords)


@typechecked
def hdbscan_cluster(
    fp: FingerprintResult,
    min_cluster_size: int = 30,
    min_samples: int | None = None,
) -> tuple[np.ndarray, int]:
    """HDBSCAN cluster labels in the original feature space.

    Returns (labels, n_clusters_excluding_noise). HDBSCAN labels noise points
    as -1.

    Notes on metric:
        HDBSCAN supports 'jaccard' and 'cosine' indirectly via precomputed
        distance matrices, but for our sizes we let it use 'hamming' for
        binary (equivalent ranking to jaccard for fixed-length bit vectors)
        and 'euclidean' on l2-normalized vectors for continuous (equivalent
        ranking to cosine).
    """
    arr = fp.array
    if fp.kind == "binary":
        arr = arr.astype(np.uint8)
        metric = "hamming"
    else:
        # l2-normalize so euclidean ~ cosine ranking
        arr = arr.astype(np.float32, copy=True)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        metric = "euclidean"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(arr)
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum())
    logger.info(
        f"HDBSCAN: {fp.name} -> {n_clusters} clusters, "
        f"{n_noise}/{len(labels)} noise points"
    )
    return labels, n_clusters


@typechecked
def embed_all(
    fps: dict[str, FingerprintResult],
    do_clustering: bool = True,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    min_cluster_size: int = 30,
    random_state: int = 0,
) -> dict[str, EmbeddingResult]:
    """Fit UMAP (and optionally HDBSCAN) for every fingerprint in the dict.

    Returns a dict keyed by short id (matching the input dict keys).
    """
    out: dict[str, EmbeddingResult] = {}
    for sid, fp in fps.items():
        coords = umap_embed(
            fp,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        if do_clustering:
            labels, k = hdbscan_cluster(fp, min_cluster_size=min_cluster_size)
        else:
            labels, k = None, 0
        out[sid] = EmbeddingResult(
            name=fp.name,
            coords=coords,
            cluster_labels=labels,
            n_clusters=k,
            metric=default_metric_for(fp.kind),
        )
    return out
