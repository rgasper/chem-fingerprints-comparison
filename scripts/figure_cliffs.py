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
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rdkit.Chem import MolFromSmiles

from fingerprints.clustering.cliffs import (
    DEFAULT_CLIFF_THRESHOLD,
    CliffRMSEResult,
    FalseFriendResult,
    cliff_knn_rmse_for_all,
    false_friend_rate_for_all,
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
    plot_cliff_rmse_bars,
    plot_false_friend_bars,
    plot_false_friend_summary,
    plot_neighbor_dy_violins,
)


CACHE_MOLACE = Path(".cache/molace")
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


def main(k: int, cliff_threshold: float) -> None:
    device = _device()
    logger.info(f"using device={device}")

    chemeleon = CheMeleonFingerprint(device=device)
    mist_28m = MISTFingerprint(model_id=MIST_28M, device=device)

    ff_by_dataset: dict[str, dict[str, FalseFriendResult]] = {}
    rmse_by_dataset: dict[str, dict[str, CliffRMSEResult]] = {}

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="neighbors per query")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_CLIFF_THRESHOLD,
        help="|Delta y| (in pKi units) above which a neighbor is a cliff",
    )
    args = parser.parse_args()
    main(k=args.k, cliff_threshold=args.threshold)
