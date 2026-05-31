"""Activity-cliff probes: how well does a fingerprint avoid 'false friends'?

Two complementary lenses, both designed to be fingerprint-symmetric (no
fingerprint plays a privileged role in defining the comparison):

- **False-friend rate at top-k**: for each molecule, the fingerprint's k
  nearest neighbors define its 'similar set'. A neighbor pair is a *false
  friend* if it is FP-similar (in the top-k) but activity-different
  (|Delta y| >= threshold). The false-friend rate is the fraction of all
  top-k pairs that are false friends. Each fingerprint is judged on its
  own neighborhood; nothing depends on Morgan or any external definition.

- **kNN cliff RMSE**: classical kNN regressor (k=5) trained on the
  MoleculeACE 'train' split, evaluated on 'test'. Test molecules are
  partitioned into cliff vs non-cliff using MoleculeACE's per-molecule
  flag, and RMSE is reported per partition. The 'cliff penalty' is
  rmse_cliff - rmse_noncliff: a fingerprint that smooths over cliffs has
  a large penalty.

The two probes ask different questions:
- false-friend rate: 'within the molecules this FP thinks are similar,
  how often is it wrong about activity?'
- cliff RMSE: 'when this FP is used to predict activity by averaging
  neighbors, how badly does it miss on cliffs?'
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from typeguard import typechecked

from fingerprints.fingerprint_methods.base import (
    FingerprintResult,
    default_metric_for,
)


# Default cliff threshold in pKi/pEC50 units. |Delta y| >= 1.0 corresponds
# to >= 10-fold potency difference, which is the canonical
# medicinal-chemistry "cliff" threshold used by MoleculeACE and others.
DEFAULT_CLIFF_THRESHOLD = 1.0


@dataclass(frozen=True)
class FalseFriendResult:
    """Top-k neighbor false-friend metrics for one fingerprint.

    Attributes:
        name: fingerprint display name
        k: neighbors per query
        cliff_threshold: |Delta y| threshold for calling a neighbor pair a
            false friend
        n_pairs: total number of (query, neighbor) pairs evaluated
        n_false_friends: count of pairs with |Delta y| >= cliff_threshold
        false_friend_rate: n_false_friends / n_pairs
        delta_y: 1D array of |Delta y| for every neighbor pair, length n_pairs
            (used for the violin plot)
    """

    name: str
    k: int
    cliff_threshold: float
    n_pairs: int
    n_false_friends: int
    false_friend_rate: float
    delta_y: np.ndarray


@dataclass(frozen=True)
class CliffRMSEResult:
    """kNN cliff vs non-cliff RMSE for one fingerprint.

    Attributes:
        name: fingerprint display name
        k: neighbors used in the kNN regressor
        rmse_all: RMSE on all test molecules
        rmse_cliff: RMSE on test molecules flagged as cliff
        rmse_noncliff: RMSE on test molecules NOT flagged as cliff
        n_test_cliff: count of cliff test mols
        n_test_noncliff: count of non-cliff test mols
    """

    name: str
    k: int
    rmse_all: float
    rmse_cliff: float
    rmse_noncliff: float
    n_test_cliff: int
    n_test_noncliff: int

    @property
    def cliff_penalty(self) -> float:
        """RMSE inflation on cliff vs non-cliff test molecules."""
        return self.rmse_cliff - self.rmse_noncliff


def _prepare_array(fp: FingerprintResult) -> tuple[np.ndarray, str]:
    """Cast for sklearn neighbors with metric matching fp.kind."""
    metric = default_metric_for(fp.kind)
    if fp.kind == "binary":
        arr = fp.array.astype(bool)
    else:
        arr = fp.array.astype(np.float32, copy=False)
    return arr, metric


@typechecked
def false_friend_rate(
    fp: FingerprintResult,
    y: np.ndarray,
    k: int = 5,
    cliff_threshold: float = DEFAULT_CLIFF_THRESHOLD,
) -> FalseFriendResult:
    """Top-k nearest-neighbor false-friend rate for one fingerprint.

    For each molecule, we take its k nearest neighbors in fingerprint space
    (excluding itself). For every (query, neighbor) pair we record |y_q - y_n|.
    A pair is a 'false friend' if |Delta y| >= cliff_threshold.

    Args:
        fp: fingerprint result over n molecules.
        y: parallel array of activity values, shape (n,). Should be in pKi /
            pEC50 units so cliff_threshold has its standard meaning.
        k: neighbors per query.
        cliff_threshold: |Delta y| threshold for false-friend classification.

    Returns:
        FalseFriendResult with the rate and the full distribution of |Delta y|.
    """
    arr, metric = _prepare_array(fp)
    n = arr.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows, fp has {n}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(arr)
    _, idx = nn.kneighbors(arr)
    idx = idx[:, 1:]  # drop self

    # delta_y has shape (n, k); flatten so each row is one (query, neighbor) pair
    delta_y = np.abs(y[:, None] - y[idx])
    delta_y_flat = delta_y.reshape(-1)
    n_pairs = int(delta_y_flat.shape[0])
    n_ff = int((delta_y_flat >= cliff_threshold).sum())
    rate = n_ff / n_pairs if n_pairs > 0 else 0.0
    logger.info(
        f"false-friend: {fp.name} k={k} cliff>={cliff_threshold} -> "
        f"{n_ff}/{n_pairs} = {rate:.3f}"
    )
    return FalseFriendResult(
        name=fp.name,
        k=k,
        cliff_threshold=cliff_threshold,
        n_pairs=n_pairs,
        n_false_friends=n_ff,
        false_friend_rate=rate,
        delta_y=delta_y_flat,
    )


@typechecked
def false_friend_rate_for_all(
    fps: dict[str, FingerprintResult],
    y: np.ndarray,
    k: int = 5,
    cliff_threshold: float = DEFAULT_CLIFF_THRESHOLD,
) -> dict[str, FalseFriendResult]:
    return {
        sid: false_friend_rate(fp, y, k=k, cliff_threshold=cliff_threshold)
        for sid, fp in fps.items()
    }


@dataclass(frozen=True)
class FalseFriendExample:
    """A single concrete false-friend pair selected as a worst-case illustration.

    Attributes:
        name: fingerprint display name
        i: dataset index of the query molecule
        j: dataset index of the neighbor molecule
        delta_y: |y_i - y_j| in pKi units
        similarity: fingerprint similarity for this pair
            (Tanimoto for binary, cosine for continuous, both higher = more similar)
        neighbor_rank: how close j was in i's neighbor ordering, in [1, k]
            (1 means j was i's nearest neighbor, k means it was kth)
    """

    name: str
    i: int
    j: int
    delta_y: float
    similarity: float
    neighbor_rank: int


@typechecked
def worst_false_friend(
    fp: FingerprintResult,
    y: np.ndarray,
    k: int = 5,
) -> FalseFriendExample:
    """Find the single top-k neighbor pair with the largest |Delta y|.

    For each molecule i we examine its k nearest neighbors in fingerprint
    space; among the resulting (i, j) pairs we return the one where the
    neighbor disagrees most strongly with i in activity. This is the
    fingerprint's worst-case "I thought these were similar but they
    aren't" example.
    """
    arr, metric = _prepare_array(fp)
    n = arr.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows, fp has {n}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(arr)
    dists, idx = nn.kneighbors(arr)
    dists = dists[:, 1:]  # drop self
    idx = idx[:, 1:]

    delta_y = np.abs(y[:, None] - y[idx])  # (n, k)
    flat_pos = int(np.argmax(delta_y))
    i = flat_pos // k
    rank = flat_pos % k  # 0-based, 0 = nearest neighbor
    j = int(idx[i, rank])
    d = float(dists[i, rank])
    # Convert distance back to a similarity in [0, 1] for display.
    # binary -> jaccard distance d, similarity = 1 - d (Tanimoto)
    # continuous -> cosine distance d, similarity = 1 - d (cosine sim in [-1, 1])
    similarity = 1.0 - d
    return FalseFriendExample(
        name=fp.name,
        i=i,
        j=j,
        delta_y=float(delta_y[i, rank]),
        similarity=similarity,
        neighbor_rank=rank + 1,
    )


@typechecked
def worst_false_friends_for_all(
    fps: dict[str, FingerprintResult],
    y: np.ndarray,
    k: int = 5,
) -> dict[str, FalseFriendExample]:
    return {sid: worst_false_friend(fp, y, k=k) for sid, fp in fps.items()}


@typechecked
def cliff_knn_rmse(
    fp: FingerprintResult,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    cliff_test_mask: np.ndarray,
    k: int = 5,
) -> CliffRMSEResult:
    """kNN regressor RMSE on cliff vs non-cliff test molecules.

    Args:
        fp: fingerprint result over n molecules in some canonical order.
        y: parallel activity values, shape (n,).
        train_mask: boolean mask of which molecules are in the train split.
        test_mask: boolean mask of which molecules are in the test split.
            train_mask and test_mask must be disjoint, and ideally cover
            every molecule.
        cliff_test_mask: boolean mask same length as y, True where the
            molecule is both in test AND flagged as a cliff molecule. The
            partition non-cliff-test = test_mask & ~cliff_test_mask.
        k: neighbors used in the kNN regressor.

    Returns:
        CliffRMSEResult with the three RMSE values.
    """
    arr, metric = _prepare_array(fp)
    n = arr.shape[0]
    for name, m in (
        ("y", y), ("train_mask", train_mask), ("test_mask", test_mask),
        ("cliff_test_mask", cliff_test_mask),
    ):
        if m.shape[0] != n:
            raise ValueError(f"{name} has {m.shape[0]} rows, fp has {n}")
    if (train_mask & test_mask).any():
        raise ValueError("train_mask and test_mask overlap")
    if (cliff_test_mask & ~test_mask).any():
        raise ValueError("cliff_test_mask must be a subset of test_mask")

    knn = KNeighborsRegressor(n_neighbors=k, metric=metric)
    knn.fit(arr[train_mask], y[train_mask])
    y_pred_test = knn.predict(arr[test_mask])
    y_true_test = y[test_mask]

    # Re-derive cliff/non-cliff sub-masks within the test ordering
    cliff_within_test = cliff_test_mask[test_mask]
    noncliff_within_test = ~cliff_within_test

    rmse_all = float(
        np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    )
    rmse_cliff = (
        float(np.sqrt(mean_squared_error(
            y_true_test[cliff_within_test], y_pred_test[cliff_within_test]
        )))
        if cliff_within_test.any()
        else float("nan")
    )
    rmse_noncliff = (
        float(np.sqrt(mean_squared_error(
            y_true_test[noncliff_within_test],
            y_pred_test[noncliff_within_test],
        )))
        if noncliff_within_test.any()
        else float("nan")
    )

    logger.info(
        f"cliff RMSE: {fp.name} k={k} -> all={rmse_all:.3f} "
        f"cliff={rmse_cliff:.3f} noncliff={rmse_noncliff:.3f}"
    )
    return CliffRMSEResult(
        name=fp.name,
        k=k,
        rmse_all=rmse_all,
        rmse_cliff=rmse_cliff,
        rmse_noncliff=rmse_noncliff,
        n_test_cliff=int(cliff_within_test.sum()),
        n_test_noncliff=int(noncliff_within_test.sum()),
    )


@typechecked
def cliff_knn_rmse_for_all(
    fps: dict[str, FingerprintResult],
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    cliff_test_mask: np.ndarray,
    k: int = 5,
) -> dict[str, CliffRMSEResult]:
    return {
        sid: cliff_knn_rmse(
            fp, y, train_mask, test_mask, cliff_test_mask, k=k,
        )
        for sid, fp in fps.items()
    }
