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


# ---------------------------------------------------------------------------
# MCS-defined cliff catches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCSCliffCandidate:
    """One verified cliff pair (i, j).

    Attributes:
        i, j: dataset indices of the two molecules (i < j).
        mcs_atoms: number of atoms in the maximum common substructure.
        mcs_fraction: mcs_atoms / min(heavy_atoms_i, heavy_atoms_j); the
            fraction of the smaller molecule covered by the MCS.
        delta_y: |y_i - y_j| in pKi / pEC50 units.
        shared_scaffold: the canonical Bemis-Murcko scaffold SMILES that
            both molecules share. Empty string if the molecules have no
            ring system (the candidate set excludes acyclic molecules).
    """

    i: int
    j: int
    mcs_atoms: int
    mcs_fraction: float
    delta_y: float
    shared_scaffold: str


def _mcs_check_one(
    mol_i, mol_j, ha_min: int, mcs_min_fraction: float, timeout: int,
) -> tuple[int, float] | None:
    """Worker function for parallel MCS verification.

    Returns (mcs_atoms, mcs_fraction) if the pair passes the threshold,
    None otherwise. Defined at module scope so joblib can pickle it.
    """
    from rdkit.Chem import rdFMCS

    res = rdFMCS.FindMCS(
        [mol_i, mol_j], timeout=timeout, completeRingsOnly=True,
    )
    if res.canceled or ha_min == 0:
        return None
    frac = res.numAtoms / ha_min
    if frac < mcs_min_fraction:
        return None
    return int(res.numAtoms), float(frac)


@typechecked
def find_mcs_cliff_candidates(
    mols: list,
    y: np.ndarray,
    cliff_threshold: float = 2.0,
    mcs_min_fraction: float = 0.7,
    mcs_timeout: int = 2,
    n_jobs: int = -1,
) -> list[MCSCliffCandidate]:
    """Build the MCS-verified cliff candidate set for one dataset.

    Pipeline (each stage prunes hard before the next):

    1. **Activity-gap filter**: keep pairs (i, j) with |y_i - y_j| >= cliff_threshold.
    2. **Same-scaffold filter**: keep pairs where both molecules have the
       same Bemis-Murcko scaffold. Acyclic molecules are excluded.
    3. **Heavy-atom compatibility**: keep pairs whose heavy-atom counts
       are within 60% of each other.
    4. **MCS verification (parallel)**: keep pairs where MCS atoms /
       min(heavy_atoms) >= `mcs_min_fraction`.

    The whole construction is fingerprint-agnostic: same-scaffold + MCS
    are pure graph properties of the molecules. This means the candidate
    set is a fair external test bed for asking "which fingerprints catch
    these cliffs?" without privileging any one fingerprint's similarity
    notion.

    Returns the list of verified candidates.
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    from joblib import Parallel, delayed

    n = len(mols)
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows, mols has {n}")

    # Stage 1: activity-gap filter (vectorized)
    n_heavy = np.array([m.GetNumHeavyAtoms() for m in mols])
    delta_y_full = np.abs(y[:, None] - y[None, :])
    i_arr, j_arr = np.where(delta_y_full >= cliff_threshold)
    mask = i_arr < j_arr
    i_arr, j_arr = i_arr[mask], j_arr[mask]
    logger.info(f"cliff candidates: |\u0394y|>={cliff_threshold} -> {len(i_arr)} pairs")

    # Stage 2: same-scaffold filter
    scaffolds: list[str] = []
    for m in mols:
        try:
            s = MurckoScaffoldSmiles(mol=m, includeChirality=False) or ""
        except (RuntimeError, ValueError):
            s = ""
        scaffolds.append(s)
    scaffolds_arr = np.array(scaffolds)
    same_scaff = (
        (scaffolds_arr[i_arr] == scaffolds_arr[j_arr])
        & (scaffolds_arr[i_arr] != "")
    )
    i_arr, j_arr = i_arr[same_scaff], j_arr[same_scaff]
    logger.info(f"  + same Bemis-Murcko scaffold -> {len(i_arr)} pairs")

    # Stage 3: heavy-atom compatibility
    ha_min = np.minimum(n_heavy[i_arr], n_heavy[j_arr])
    ha_max = np.maximum(n_heavy[i_arr], n_heavy[j_arr])
    mask = ha_max > 0
    mask &= (ha_min / np.maximum(ha_max, 1)) >= 0.6
    i_arr, j_arr = i_arr[mask], j_arr[mask]
    ha_min = ha_min[mask]
    logger.info(f"  + heavy-atom compatibility -> {len(i_arr)} pairs")

    if len(i_arr) == 0:
        return []

    # Stage 4: parallel MCS verification
    args = [
        (mols[int(i)], mols[int(j)], int(ha_min[k]), mcs_min_fraction, mcs_timeout)
        for k, (i, j) in enumerate(zip(i_arr, j_arr))
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_mcs_check_one)(*a) for a in args
    )

    verified: list[MCSCliffCandidate] = []
    for k, r in enumerate(results):
        if r is None:
            continue
        mcs_atoms, frac = r
        i, j = int(i_arr[k]), int(j_arr[k])
        verified.append(
            MCSCliffCandidate(
                i=i, j=j,
                mcs_atoms=mcs_atoms,
                mcs_fraction=frac,
                delta_y=float(abs(y[i] - y[j])),
                shared_scaffold=scaffolds[i],
            )
        )
    logger.info(
        f"  + MCS verified (>= {mcs_min_fraction} fraction) -> {len(verified)}"
    )
    return verified


@dataclass(frozen=True)
class BestCatchExample:
    """A single concrete cliff catch as a best-case illustration.

    Attributes:
        name: fingerprint display name
        i, j: dataset indices of the two molecules.
        delta_y: |y_i - y_j|
        similarity: fingerprint similarity for this pair
        mcs_fraction: structural similarity (MCS atoms / min heavy atoms)
        rank_among_mol_i: where j sits in i's neighbor ordering (1 = nearest);
            the catch test is satisfied iff this rank > k AND the symmetric
            rank (where i sits in j's order) > k.
    """

    name: str
    i: int
    j: int
    delta_y: float
    similarity: float
    mcs_fraction: float
    rank_among_mol_i: int


@typechecked
def best_cliff_catch(
    fp: FingerprintResult,
    candidates: list[MCSCliffCandidate],
    k: int = 5,
) -> BestCatchExample | None:
    """Among MCS-verified cliff pairs, the best catch for this fingerprint.

    'Best' = the pair the FP would be most expected to put into top-k (so
    rejecting it is the most impressive catch). Concretely: among pairs
    the FP correctly excluded from both molecules' top-k neighbor sets,
    pick the one with the highest FP similarity. That answers "what's
    the closest call this FP correctly avoided?" rather than "what's
    the biggest cliff this FP happened to catch?", which would surface
    the same trivial example across all fingerprints.

    Returns None if the fingerprint failed to catch any candidate.
    """
    if not candidates:
        return None
    arr, metric = _prepare_array(fp)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(arr)
    _, idx = nn.kneighbors(arr)
    idx_no_self = idx[:, 1:]

    best: tuple[float, MCSCliffCandidate, float] | None = None  # (sim, cand, dist)
    for cand in candidates:
        i, j = cand.i, cand.j
        i_topk = set(idx_no_self[i].tolist())
        j_topk = set(idx_no_self[j].tolist())
        if j in i_topk or i in j_topk:
            continue
        # Compute pair distance / similarity
        if fp.kind == "binary":
            a, b = arr[i], arr[j]
            inter = int(np.sum(a & b))
            union = int(np.sum(a | b))
            d = 0.0 if union == 0 else 1.0 - inter / union
        else:
            from sklearn.metrics.pairwise import paired_distances
            d = float(paired_distances(arr[i:i+1], arr[j:j+1], metric=metric)[0])
        sim = 1.0 - d
        if best is None or sim > best[0]:
            best = (sim, cand, d)

    if best is None:
        return None
    sim, cand, _ = best
    return BestCatchExample(
        name=fp.name,
        i=cand.i, j=cand.j,
        delta_y=cand.delta_y,
        similarity=sim,
        mcs_fraction=cand.mcs_fraction,
        rank_among_mol_i=k + 1,
    )


@typechecked
def best_cliff_catches_for_all(
    fps: dict[str, FingerprintResult],
    candidates: list[MCSCliffCandidate],
    k: int = 5,
) -> dict[str, BestCatchExample | None]:
    return {sid: best_cliff_catch(fp, candidates, k=k) for sid, fp in fps.items()}


@dataclass(frozen=True)
class CatchRateResult:
    """Aggregate catch-rate statistics for one fingerprint over a candidate set.

    Attributes:
        name: fingerprint display name
        n_candidates: total MCS-verified cliff candidates considered
        n_caught: count of candidates this FP placed outside both members'
            top-k neighbor sets
        catch_rate: n_caught / n_candidates
        mean_delta_y_caught: average |Delta y| of caught candidates
            (NaN if n_caught == 0)
    """

    name: str
    n_candidates: int
    n_caught: int
    catch_rate: float
    mean_delta_y_caught: float


@typechecked
def cliff_catch_rate(
    fp: FingerprintResult,
    candidates: list[MCSCliffCandidate],
    k: int = 5,
) -> CatchRateResult:
    """Fraction of candidate cliffs the fingerprint correctly avoids in its
    top-k neighborhoods (catches).
    """
    if not candidates:
        return CatchRateResult(
            name=fp.name, n_candidates=0, n_caught=0,
            catch_rate=0.0, mean_delta_y_caught=float("nan"),
        )
    arr, metric = _prepare_array(fp)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(arr)
    _, idx = nn.kneighbors(arr)
    idx_no_self = idx[:, 1:]

    n_caught = 0
    sum_dy = 0.0
    for cand in candidates:
        i_topk = set(idx_no_self[cand.i].tolist())
        j_topk = set(idx_no_self[cand.j].tolist())
        if cand.j not in i_topk and cand.i not in j_topk:
            n_caught += 1
            sum_dy += cand.delta_y
    rate = n_caught / len(candidates)
    mean_dy = sum_dy / n_caught if n_caught > 0 else float("nan")
    logger.info(
        f"catch rate: {fp.name} k={k} -> {n_caught}/{len(candidates)} = "
        f"{rate:.3f} (mean |\u0394y| caught = {mean_dy:.2f})"
    )
    return CatchRateResult(
        name=fp.name,
        n_candidates=len(candidates),
        n_caught=n_caught,
        catch_rate=rate,
        mean_delta_y_caught=mean_dy,
    )


@typechecked
def cliff_catch_rate_for_all(
    fps: dict[str, FingerprintResult],
    candidates: list[MCSCliffCandidate],
    k: int = 5,
) -> dict[str, CatchRateResult]:
    return {sid: cliff_catch_rate(fp, candidates, k=k) for sid, fp in fps.items()}


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
