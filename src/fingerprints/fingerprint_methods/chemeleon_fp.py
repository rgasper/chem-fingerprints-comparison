"""CheMeleon learned fingerprint wrapper.

Lifts CheMeleon's MPNN encoder (trained to predict Mordred descriptors) into
the same FingerprintResult interface used by the classical fingerprints.

Weights are downloaded from Zenodo on first use and cached under ~/.chemprop.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from loguru import logger
from rdkit.Chem import Mol

from fingerprints.exceptions import ModelLoadError
from fingerprints.fingerprint_methods.base import FingerprintResult


CHEMELEON_WEIGHTS_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


class CheMeleonFingerprint:
    """Wraps the CheMeleon MPNN encoder. Stateful: keeps the model loaded.

    Example:
        >>> fp = CheMeleonFingerprint(device='mps')
        >>> result = fp([Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('c1ccccc1')])
        >>> result.array.shape
        (2, 2048)
    """

    name = "chemeleon"

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        ckpt_dir = Path.home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not mp_path.exists():
            logger.info(f"downloading chemeleon weights to {mp_path}")
            try:
                urlretrieve(CHEMELEON_WEIGHTS_URL, mp_path)
            except Exception as e:
                raise ModelLoadError(
                    f"failed to fetch chemeleon weights from {CHEMELEON_WEIGHTS_URL}"
                ) from e

        ckpt = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
        agg = nn.MeanAggregation()
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)
        self.n_features = mp.output_dim

    def __call__(self, mols: list[Mol]) -> FingerprintResult:
        bmg = BatchMolGraph([self.featurizer(m) for m in mols])
        bmg.to(device=self.model.device)
        with torch.no_grad():
            arr = self.model.fingerprint(bmg).numpy(force=True).astype(np.float32)
        return FingerprintResult(
            name="CheMeleon", kind="continuous", array=arr, n_features=arr.shape[1]
        )
