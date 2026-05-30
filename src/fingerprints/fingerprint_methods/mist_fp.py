"""MIST learned-fingerprint wrappers.

MIST is a family of RoBERTa-PreLayerNorm encoders pretrained with masked
language modeling on SMILES (using the smirk tokenizer). We embed a molecule
as the mean of last_hidden_state over non-pad tokens.

We expose the 28M and 1.8B pretrained variants. The 1.8B model is large
(~7GB on disk) and slow on CPU; on Apple Silicon use device='mps'.

Note on cosine similarity: raw mean-pooled MIST embeddings are strongly
anisotropic - cosine similarities collapse into a tight band ~0.85-0.97
across all molecules, regardless of how (dis)similar the molecules
actually are. This is a known phenomenon for unsupervised pretrained
transformer encoders (Ethayarajh 2019, Mu & Viswanath 2018) and it means
raw MIST outputs are not directly comparable as a similarity fingerprint
without dataset-dependent post-processing (mean centering, all-but-the-top,
etc.). We deliberately do NOT apply such post-processing here so that
"the MIST fingerprint" is a single deterministic function of the input
molecule, comparable like-for-like with the classical fingerprints.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from loguru import logger
from rdkit.Chem import Mol, MolToSmiles
from smirk import SmirkTokenizerFast
from tqdm import tqdm
from transformers import AutoModel

from fingerprints.exceptions import ModelLoadError
from fingerprints.fingerprint_methods.base import FingerprintResult


MIST_28M = "mist-models/mist-28M-ti624ev1"
MIST_1_8B = "mist-models/mist-1.8B-dh61satt"


class MISTFingerprint:
    """Wraps a pretrained MIST encoder.

    Args:
        model_id: HuggingFace MIST model id
        device: torch device (None -> cpu)
        batch_size: tokens are short so batches can be moderately large
        display_name: override the displayed name (defaults to MIST-<size>)

    Example:
        >>> fp = MISTFingerprint(model_id=MIST_28M, device='mps')
        >>> result = fp([Chem.MolFromSmiles('CCO')])
        >>> result.array.shape
        (1, 512)
    """

    def __init__(
        self,
        model_id: str = MIST_28M,
        device: str | torch.device | None = None,
        batch_size: int = 32,
        display_name: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.display_name = display_name or self._auto_name(model_id)
        self.name = self.display_name

        try:
            self.tokenizer = SmirkTokenizerFast.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            raise ModelLoadError(f"failed to load MIST {model_id}") from e
        self.model.eval()

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model.to(self.device)
        self.n_features = self.model.config.hidden_size

    @staticmethod
    def _auto_name(model_id: str) -> str:
        if "1.8B" in model_id:
            return "MIST-1.8B"
        if "28M" in model_id:
            return "MIST-28M"
        return f"MIST({model_id.split('/')[-1]})"

    def _smiles_for(self, mol: Mol | str) -> str:
        return mol if isinstance(mol, str) else MolToSmiles(mol)

    def __call__(self, mols: list[Mol] | list[str]) -> FingerprintResult:
        smiles = [self._smiles_for(m) for m in mols]
        return self.embed_smiles(smiles)

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, H), attention_mask: (B, T). Returns (B, H)."""
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def embed_smiles(self, smiles: list[str], show_progress: bool = False) -> FingerprintResult:
        all_vecs: list[np.ndarray] = []
        iterator: Iterable[int] = range(0, len(smiles), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=(len(smiles) + self.batch_size - 1) // self.batch_size,
                desc=f"{self.name} embedding",
            )
        for start in iterator:
            chunk = smiles[start : start + self.batch_size]
            batch = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                out = self.model(**batch)
            pooled = self._mean_pool(out.last_hidden_state, batch["attention_mask"])
            all_vecs.append(pooled.to(torch.float32).cpu().numpy())
        arr = np.concatenate(all_vecs, axis=0)
        logger.info(f"{self.name}: produced fingerprints of shape {arr.shape}")
        return FingerprintResult(
            name=self.name, kind="continuous", array=arr, n_features=arr.shape[1]
        )
