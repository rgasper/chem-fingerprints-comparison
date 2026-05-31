"""Generate activity-cliff figures on three MoleculeACE targets.

For each target (Dopamine D3, Thrombin, GSK-3 beta) we produce:
- 01_false_friend_rate_<target>.png: bar chart of top-k false-friend rate.
- 02_neighbor_dy_violins_<target>.png: violin distributions of |Delta y| at top-k.
- 03_cliff_rmse_<target>.png: kNN regression RMSE on cliff vs non-cliff test mols.

Plus one cross-dataset:
- 04_false_friend_summary.png: heatmap (datasets x fingerprints).

Run:
  uv run python scripts/figure_cliffs.py
  uv run python scripts/figure_cliffs.py --k 10 --threshold 1.5  # tune k / cliff threshold
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.cliffs import (
    DEFAULT_CLIFF_THRESHOLD,
    BestCatchExample,
    CatchRateResult,
    CliffRMSEResult,
    FalseFriendExample,
    FalseFriendResult,
    MCSCliffCandidate,
    best_cliff_catches_for_all,
    cliff_catch_rate_for_all,
    cliff_knn_rmse_for_all,
    false_friend_rate_for_all,
    find_mcs_cliff_candidates,
    worst_false_friends_for_all,
)
from fingerprints.data.molace import (
    D3_DOPAMINE,
    GSK3B,
    THROMBIN,
    MolACEDataset,
    load_molace,
)
from fingerprints.fingerprint_methods import rdkit_fps
from fingerprints.fingerprint_methods.base import FingerprintResult
from fingerprints.fingerprint_methods.chemeleon_fp import CheMeleonFingerprint
from fingerprints.fingerprint_methods.mist_fp import MIST_28M, MISTFingerprint
from fingerprints.plots.cliffs import (
    plot_catch_rate_summary,
    plot_cliff_rmse_bars,
    plot_false_friend_bars,
    plot_false_friend_summary,
    plot_neighbor_dy_violins,
)
from fingerprints.plots.cliff_examples import (
    plot_cliff_examples,
    plot_worst_false_friend_examples,
)


CACHE_MOLACE = Path(".cache/molace")
CACHE_CANDIDATES = Path(".cache/cliff_candidates")
FIG_DIR = Path("figures/cliffs")

DATASETS = (D3_DOPAMINE, THROMBIN, GSK3B)


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _build_fps(
    mols: list,
    chemeleon: CheMeleonFingerprint,
    mist_28m: MISTFingerprint,
) -> dict[str, FingerprintResult]:
    fps: dict[str, FingerprintResult] = dict(rdkit_fps.all_classical(mols))
    fps["chemeleon"] = chemeleon(mols)
    fps["mist_28M"] = mist_28m(mols)
    return fps


def _load_dataset(ds: MolACEDataset) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Load a MoleculeACE dataset, parse with RDKit, drop unparseable rows.

    Returns (mols, y, cliff_mol_mask, split_str_array). Index alignment:
    mols[i] corresponds to y[i], cliff_mol_mask[i], split_str_array[i].
    """
    df = load_molace(ds, CACHE_MOLACE)
    smis = df["smiles"].to_list()
    parsed = [(s, MolFromSmiles(s), i) for i, s in enumerate(smis)]
    keep = [(m, i) for s, m, i in parsed if m is not None]
    mols = [m for m, _ in keep]
    keep_idx = np.array([i for _, i in keep])
    y = df["y"].to_numpy()[keep_idx]
    cliff = df["cliff_mol"].to_numpy()[keep_idx]
    split = np.array(df["split"].to_list())[keep_idx]
    if len(mols) < df.height:
        logger.info(
            f"  parsed {len(mols)}/{df.height} mols; "
            f"dropped {df.height - len(mols)}"
        )
    return mols, y, cliff, split


def _load_or_compute_candidates(
    ds: MolACEDataset,
    mols: list,
    y: np.ndarray,
    cliff_threshold: float = 2.0,
    mcs_min_fraction: float = 0.7,
    use_cache: bool = True,
) -> list[MCSCliffCandidate]:
    """Build (or load from cache) the MCS-verified cliff candidate set."""
    cache_path = CACHE_CANDIDATES / f"{ds.name}_dy{cliff_threshold}_mcs{mcs_min_fraction}.pkl"
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            cands = pickle.load(fh)
        logger.info(f"loaded {len(cands)} cached candidates from {cache_path}")
        return cands
    cands = find_mcs_cliff_candidates(
        mols, y,
        cliff_threshold=cliff_threshold,
        mcs_min_fraction=mcs_min_fraction,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(cands, fh)
    logger.info(f"cached {len(cands)} candidates to {cache_path}")
    return cands


def main(k: int, cliff_threshold: float) -> None:
    device = _device()
    logger.info(f"using device={device}")

    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    ff_by_dataset: dict[str, dict[str, FalseFriendResult]] = {}
    rmse_by_dataset: dict[str, dict[str, CliffRMSEResult]] = {}
    catch_by_dataset: dict[str, dict[str, CatchRateResult]] = {}

    for ds in DATASETS:
        logger.info(f"=== {ds.name} ({ds.target_label}) ===")
        mols, y, cliff, split = _load_dataset(ds)
        fps = _build_fps(mols, chemeleon, mist_28m)

        ff = false_friend_rate_for_all(
            fps, y, k=k, cliff_threshold=cliff_threshold,
        )
        ff_by_dataset[ds.target_label] = ff

        train_mask = split == "train"
        test_mask = split == "test"
        cliff_test_mask = test_mask & cliff
        if not cliff_test_mask.any():
            logger.warning(
                f"{ds.name} has no cliff test molecules; skipping kNN RMSE"
            )
            rmse_by_dataset[ds.target_label] = {}
        else:
            rmse = cliff_knn_rmse_for_all(
                fps, y, train_mask, test_mask, cliff_test_mask, k=k,
            )
            rmse_by_dataset[ds.target_label] = rmse

        # MCS-verified cliff candidates (cached) and catch metrics.
        candidates = _load_or_compute_candidates(ds, mols, y)
        catches_examples: dict[str, BestCatchExample | None] = (
            best_cliff_catches_for_all(fps, candidates, k=k)
        )
        catch_rates = cliff_catch_rate_for_all(fps, candidates, k=k)
        catch_by_dataset[ds.target_label] = catch_rates

        # Per-dataset figures
        plot_false_friend_bars(
            ff,
            out_path=FIG_DIR / f"01_false_friend_rate_{ds.name}.png",
            title=(
                f"False-friend rate at top-k={k} \u2014 {ds.target_label} "
                f"({ds.target_class}, n={len(mols)})"
            ),
        )
        plot_neighbor_dy_violins(
            ff,
            out_path=FIG_DIR / f"02_neighbor_dy_violins_{ds.name}.png",
            title=(
                f"|\u0394y| within top-k={k} neighbors \u2014 {ds.target_label}"
            ),
        )
        if rmse_by_dataset[ds.target_label]:
            plot_cliff_rmse_bars(
                rmse_by_dataset[ds.target_label],
                out_path=FIG_DIR / f"03_cliff_rmse_{ds.name}.png",
                title=(
                    f"kNN regression cliff vs non-cliff RMSE \u2014 "
                    f"{ds.target_label} (k={k})"
                ),
            )

        # Worst false friend + best catch example figure (D3 only for now;
        # extend to all targets in a follow-up).
        if ds is D3_DOPAMINE:
            ff_examples = worst_false_friends_for_all(fps, y, k=k)
            plot_cliff_examples(
                ff_examples, catches_examples, mols=mols, y=y,
                out_path=FIG_DIR / f"05_cliff_examples_{ds.name}.png",
                title=(
                    f"Cliff examples per fingerprint \u2014 "
                    f"{ds.target_label} (top-k={k}, "
                    f"{len(candidates)} MCS-verified candidates)"
                ),
            )

        # Free fingerprint memory before the next dataset
        del fps

    # Cross-dataset summary
    plot_false_friend_summary(
        ff_by_dataset,
        out_path=FIG_DIR / "04_false_friend_summary.png",
        title=(
            f"False-friend rate at top-k={k}, "
            f"cliff threshold |\u0394y| \u2265 {cliff_threshold}"
        ),
    )
    plot_catch_rate_summary(
        catch_by_dataset,
        out_path=FIG_DIR / "06_catch_rate_summary.png",
        title=(
            f"MCS-cliff catch rate at top-k={k} "
            f"(same Bemis-Murcko scaffold + |\u0394y|\u22652.0 + MCS\u22650.7)"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="neighbors per query")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_CLIFF_THRESHOLD,
        help="|Delta y| (in pKi units) above which a neighbor is a cliff",
    )
    args = parser.parse_args()
    main(k=args.k, cliff_threshold=args.threshold)
