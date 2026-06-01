"""Aggregate cliff metrics: matched non-cliff controls + PR-AUC, plus an
L2-distance variant for neural fingerprints to probe whether the cliff-
blindness hierarchy is a metric artifact.

Two extensions over `cliffs.py`:

1. **Matched non-cliff control sampler.** For each cliff at graph_distance d,
   sample a non-cliff pair at the same graph_distance (within +/-1) with
   |delta y| < 1.0. This conditions away the trivial "FP can't separate
   close pairs from random pairs" baseline and isolates the question
   "given two molecules at the same structural distance, can the FP tell
   the cliff from the non-cliff?"

2. **PR-AUC of cliff vs matched non-cliff.** Each FP scores both pairs;
   we treat (cliff = positive class, non-cliff = negative class) and
   compute average precision. PR-AUC at this prevalence (50%) is interpretable:
   0.5 = random, 1.0 = perfect separation.

3. **L2 distance for neural fingerprints.** Cosine on continuous embeddings
   compresses everything into a high-baseline range. Computing L2 distance
   instead lets us check whether neural FPs really lack discriminative
   power on cliffs or whether cosine just hides it.

Note: PR-AUC is dataset-bound. The threshold required to make a binary
decision from any of these scores depends on the dataset's composition.
We report PR-AUC alongside cliff-blind-rate-at-0.7 not as "the new portable
metric" but as a complementary signal: a high PR-AUC with a high
cliff-blind rate means "the FP can rank cliffs vs non-cliffs but its
absolute scale is calibrated wrong."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import paired_distances
from typeguard import typechecked

from fingerprints.clustering.cliffs import CliffPair, _atom_composition, _composition_l1, _mcs_atoms
from fingerprints.fingerprint_methods.base import FingerprintResult


DEFAULT_NONCLIFF_DELTA_Y_MAX = 1.0
DEFAULT_DISTANCE_TOL = 1


@dataclass(frozen=True)
class NonCliffPair:
    """A non-cliff pair sampled at a target graph distance.

    Attributes:
        i, j: dataset indices, i < j.
        graph_distance: same metric as CliffPair.graph_distance.
        delta_y: |y_i - y_j|, kept for diagnostics; guaranteed < the
            non-cliff delta-y cap used at sampling time.
    """

    i: int
    j: int
    graph_distance: int
    delta_y: float


@typechecked
def sample_matched_noncliffs(
    mols: list,
    y: np.ndarray,
    cliff_pairs: list[CliffPair],
    delta_y_max: float = DEFAULT_NONCLIFF_DELTA_Y_MAX,
    distance_tol: int = DEFAULT_DISTANCE_TOL,
    seed: int = 0,
    candidate_pool_size: int = 20_000,
    n_jobs: int = -1,
    mcs_timeout: int = 2,
) -> list[NonCliffPair]:
    """For each cliff, sample one non-cliff pair at the same graph distance.

    Strategy:

    1. Build a candidate pool of pairs with |delta y| < delta_y_max,
       pre-filtered on |delta n_atoms| and composition L1 to bound
       graph_distance. Subsample to candidate_pool_size to keep MCS work
       bounded.
    2. Compute graph_distance for the entire candidate pool in parallel
       (one MCS run per candidate, joblib threads since RDKit MCS releases
       the GIL).
    3. Bucket candidates by graph_distance. For each cliff at gd = d, pick
       a random candidate from buckets {d - tol, ..., d + tol} that hasn't
       been used yet.

    This is much faster than the per-cliff walk because every candidate's
    MCS is computed exactly once and reused across all cliffs that need
    its bucket.

    Args:
        mols: molecule list (same order as cliff_pair indices).
        y: property array of length len(mols).
        cliff_pairs: cliffs whose graph_distances drive the matching.
        delta_y_max: a non-cliff has |delta y| < this.
        distance_tol: allow graph_distance within +/- this of target.
        seed: rng seed for candidate shuffling and bucket sampling.
        candidate_pool_size: cap candidate pool size after pre-filter.
            Larger = better coverage at higher MCS cost.
        n_jobs: joblib parallelism (-1 = all cores).
        mcs_timeout: per-pair MCS timeout in seconds.
    """
    rng = np.random.default_rng(seed)
    n = len(mols)
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows, mols has {n}")

    if not cliff_pairs:
        return []

    n_atoms = np.array([m.GetNumHeavyAtoms() for m in mols])
    comps = [_atom_composition(m) for m in mols]

    targets = np.array([cp.graph_distance for cp in cliff_pairs])
    max_target = int(targets.max())
    delta_atoms_cap = max_target + distance_tol
    l1_cap = max_target + distance_tol

    # Stage 1: |delta y| < delta_y_max, upper triangle.
    delta_y_full = np.abs(y[:, None] - y[None, :])
    i_arr, j_arr = np.where(delta_y_full < delta_y_max)
    mask = i_arr < j_arr
    i_arr, j_arr = i_arr[mask], j_arr[mask]
    logger.info(
        f"non-cliff candidates: |\u0394y| < {delta_y_max} -> {len(i_arr)} pairs"
    )

    # Stage 2: |delta n_atoms|
    mask = np.abs(n_atoms[i_arr] - n_atoms[j_arr]) <= delta_atoms_cap
    i_arr, j_arr = i_arr[mask], j_arr[mask]
    logger.info(f"  + |\u0394n_atoms| <= {delta_atoms_cap} -> {len(i_arr)} pairs")

    # Stage 3: composition L1
    mask = np.array([
        _composition_l1(comps[int(i)], comps[int(j)]) <= l1_cap
        for i, j in zip(i_arr, j_arr)
    ])
    i_arr, j_arr = i_arr[mask], j_arr[mask]
    logger.info(f"  + composition L1 <= {l1_cap} -> {len(i_arr)} pairs")

    if len(i_arr) == 0:
        return []

    # Stage 4: subsample candidate pool, then parallel MCS on the survivors.
    if len(i_arr) > candidate_pool_size:
        keep = rng.choice(len(i_arr), size=candidate_pool_size, replace=False)
        i_arr = i_arr[keep]
        j_arr = j_arr[keep]
        logger.info(
            f"  subsampled to candidate_pool_size={candidate_pool_size}"
        )

    logger.info(
        f"  running parallel MCS on {len(i_arr)} candidates (n_jobs={n_jobs})"
    )
    mcs_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_mcs_atoms)(mols[int(i)], mols[int(j)], mcs_timeout)
        for i, j in zip(i_arr, j_arr)
    )

    # Bucket candidates by graph_distance.
    buckets: dict[int, list[int]] = {}
    for k, mcs in enumerate(mcs_results):
        if mcs is None:
            continue
        i, j = int(i_arr[k]), int(j_arr[k])
        gd = int(n_atoms[i]) + int(n_atoms[j]) - 2 * mcs
        buckets.setdefault(gd, []).append(k)

    bucket_sizes = {d: len(v) for d, v in sorted(buckets.items())}
    logger.info(f"  graph_distance buckets: {bucket_sizes}")

    # Shuffle each bucket once so per-cliff sampling is uniform.
    for d in buckets:
        rng.shuffle(buckets[d])

    # Per-cliff: pull one un-used candidate from buckets in [d - tol, d + tol].
    sampled: list[NonCliffPair] = []
    used: set[tuple[int, int]] = set()
    bucket_cursors: dict[int, int] = {d: 0 for d in buckets}

    for cp in cliff_pairs:
        target = int(cp.graph_distance)
        accepted: NonCliffPair | None = None
        # Walk bands outward: target first, then target-1, target+1, ...
        bands = [target]
        for off in range(1, distance_tol + 1):
            bands.extend([target - off, target + off])
        for d in bands:
            if d not in buckets:
                continue
            while bucket_cursors[d] < len(buckets[d]):
                cand_k = buckets[d][bucket_cursors[d]]
                bucket_cursors[d] += 1
                i = int(i_arr[cand_k])
                j = int(j_arr[cand_k])
                if (i, j) in used:
                    continue
                used.add((i, j))
                accepted = NonCliffPair(
                    i=i, j=j, graph_distance=d,
                    delta_y=float(abs(y[i] - y[j])),
                )
                break
            if accepted is not None:
                break
        if accepted is not None:
            sampled.append(accepted)
        else:
            logger.debug(
                f"no non-cliff match for cliff ({cp.i}, {cp.j}) gd={target}"
            )

    logger.info(
        f"matched non-cliffs: {len(sampled)}/{len(cliff_pairs)} cliffs covered"
    )
    return sampled


def _pair_distances(
    fp: FingerprintResult,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Pairwise distances under the requested metric.

    Args:
        fp: fingerprint
        pairs_i, pairs_j: index arrays into fp.array
        metric: 'jaccard' | 'cosine' | 'euclidean'
    """
    arr = fp.array
    if metric == "jaccard":
        a = arr[pairs_i].astype(bool)
        b = arr[pairs_j].astype(bool)
        intersection = np.sum(a & b, axis=1)
        union = np.sum(a | b, axis=1)
        sim = np.where(union == 0, 0.0, intersection / np.maximum(union, 1))
        return 1.0 - sim
    arr_f = arr.astype(np.float32, copy=False)
    a = arr_f[pairs_i]
    b = arr_f[pairs_j]
    return paired_distances(a, b, metric=metric)


@dataclass(frozen=True)
class CliffSeparationResult:
    """How well the FP separates cliffs from matched non-cliffs.

    Attributes:
        name: fingerprint display name.
        metric: distance metric used ('jaccard' | 'cosine' | 'euclidean').
        n_cliffs: number of cliff pairs scored.
        n_noncliffs: number of matched non-cliff pairs scored.
        cliff_sims: similarity values on cliff pairs (1 - distance).
        noncliff_sims: similarity values on non-cliff pairs.
        median_cliff_sim: median of cliff_sims.
        median_noncliff_sim: median of noncliff_sims.
        cliff_blind_rate_07: P(cliff_sim >= 0.7), inherits the original
            scale-dependent threshold; reported for continuity with the
            existing summary heatmap. Meaningful only for jaccard / cosine,
            not euclidean (where similarity = 1 - distance can be negative
            and 0.7 is not a calibrated cutoff).
        pr_auc: average precision treating cliffs as the positive class
            and non-cliffs as the negative class. Higher = better
            separation. 0.5 = random under 1:1 prevalence.
    """

    name: str
    metric: str
    n_cliffs: int
    n_noncliffs: int
    cliff_sims: np.ndarray
    noncliff_sims: np.ndarray
    median_cliff_sim: float
    median_noncliff_sim: float
    cliff_blind_rate_07: float
    pr_auc: float


@typechecked
def cliff_separation_for(
    fp: FingerprintResult,
    cliff_pairs: list[CliffPair],
    noncliff_pairs: list[NonCliffPair],
    metric: str | None = None,
) -> CliffSeparationResult:
    """Compute cliff-vs-non-cliff separation metrics for one FP.

    For PR-AUC, we score each pair by similarity = 1 - distance and label
    cliffs = 0, non-cliffs = 1. The intuition is "cliffs SHOULD score
    LOW similarity"; an FP that does this perfectly will rank non-cliffs
    above cliffs, giving high average precision. We use this orientation
    so the metric reads "higher = better cliff awareness."

    For metric selection: when `metric` is None, defaults from
    `default_metric_for(fp.kind)` (jaccard for binary, cosine for
    continuous). Pass 'euclidean' to override on continuous FPs.
    """
    from fingerprints.fingerprint_methods.base import default_metric_for

    if metric is None:
        metric = default_metric_for(fp.kind)
    if not cliff_pairs:
        raise ValueError("cliff_pairs is empty")
    if not noncliff_pairs:
        raise ValueError("noncliff_pairs is empty")

    cliff_i = np.array([p.i for p in cliff_pairs])
    cliff_j = np.array([p.j for p in cliff_pairs])
    nc_i = np.array([p.i for p in noncliff_pairs])
    nc_j = np.array([p.j for p in noncliff_pairs])

    cliff_d = _pair_distances(fp, cliff_i, cliff_j, metric)
    nc_d = _pair_distances(fp, nc_i, nc_j, metric)

    # Use 1 - distance as the similarity-like score. For jaccard / cosine
    # this is in [0, 1]; for euclidean it can be negative. PR-AUC only
    # cares about the ranking, so the absolute scale is fine - but we
    # still suppress cliff_blind_rate_07 for non-bounded metrics.
    cliff_sims = 1.0 - cliff_d
    noncliff_sims = 1.0 - nc_d

    # PR-AUC: positive class = non-cliff (we want non-cliffs ranked HIGHER
    # in similarity than cliffs, since "non-cliff means the molecules
    # really are similar"). A perfect FP would give every non-cliff a
    # higher similarity than every cliff, yielding average precision = 1.
    y_true = np.concatenate([
        np.zeros(cliff_sims.shape[0]),  # cliffs = 0
        np.ones(noncliff_sims.shape[0]),  # non-cliffs = 1
    ])
    scores = np.concatenate([cliff_sims, noncliff_sims])
    pr_auc = float(average_precision_score(y_true, scores))

    median_cliff = float(np.median(cliff_sims))
    median_nc = float(np.median(noncliff_sims))
    if metric in ("jaccard", "cosine"):
        cb07 = float((cliff_sims >= 0.7).mean())
    else:
        cb07 = float("nan")

    logger.info(
        f"separation: {fp.name} metric={metric} "
        f"med_cliff={median_cliff:.3f} med_nc={median_nc:.3f} "
        f"PR-AUC={pr_auc:.3f}"
    )

    return CliffSeparationResult(
        name=fp.name,
        metric=metric,
        n_cliffs=cliff_sims.shape[0],
        n_noncliffs=noncliff_sims.shape[0],
        cliff_sims=cliff_sims,
        noncliff_sims=noncliff_sims,
        median_cliff_sim=median_cliff,
        median_noncliff_sim=median_nc,
        cliff_blind_rate_07=cb07,
        pr_auc=pr_auc,
    )


@typechecked
def cliff_separation_for_all(
    fps: dict[str, FingerprintResult],
    cliff_pairs: list[CliffPair],
    noncliff_pairs: list[NonCliffPair],
    extra_metrics: dict[str, str] | None = None,
) -> dict[str, CliffSeparationResult]:
    """Run cliff_separation_for over every FP, with optional metric override.

    Args:
        fps: dict short_id -> FingerprintResult.
        cliff_pairs: cliff list.
        noncliff_pairs: matched non-cliff list.
        extra_metrics: optional {short_id: metric} mapping that overrides
            the default metric for those FPs. Used for the L2 sanity
            check on neural FPs - pass {'chemeleon': 'euclidean',
            'mist_28M': 'euclidean'} to get the alternative metric.
    """
    extra_metrics = extra_metrics or {}
    out: dict[str, CliffSeparationResult] = {}
    for sid, fp in fps.items():
        metric = extra_metrics.get(sid)
        out[sid] = cliff_separation_for(fp, cliff_pairs, noncliff_pairs, metric=metric)
    return out
