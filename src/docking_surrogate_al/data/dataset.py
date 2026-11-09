"""Dataset assembly and dataloaders.

Design note
-----------
The molecule pool is small enough (5k to 50k graphs) to featurise once and
hold in memory, so there is no on-disk graph cache to invalidate. Featurising
is done once per process and reused across active-learning rounds, which is
where it pays off: ten rounds times three strategies times five seeds would
otherwise re-featurise the same pool 150 times.

Targets are standardised (zero mean, unit variance) using statistics computed
on the *training* partition only, and the scaler travels with the checkpoint.
Computing them over the whole pool would leak test-set distribution
information into training, which is a small leak but a real one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..config import TrainConfig
from ..seeds import split_generator, worker_init_fn
from .featurise import mol_to_graph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetScaler:
    """Affine target scaling fitted on the training partition only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> TargetScaler:
        """Fit to ``values``, guarding against a degenerate zero variance."""
        std = float(np.std(values))
        if std < 1e-8:
            logger.warning("Target standard deviation is ~0; scaling disabled.")
            std = 1.0
        return cls(mean=float(np.mean(values)), std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Scale raw docking scores to the model's target space."""
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Map model outputs back to kcal/mol."""
        return values * self.std + self.mean

    def inverse_std(self, values: np.ndarray) -> np.ndarray:
        """Map a predictive standard deviation back to kcal/mol.

        Only the scale applies to a spread, not the offset. Getting this wrong
        is how calibration plots end up silently shifted.
        """
        return values * self.std

    def to_dict(self) -> dict[str, float]:
        """Return a checkpoint-serialisable mapping."""
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> TargetScaler:
        """Rebuild from a checkpoint payload."""
        return cls(mean=float(payload["mean"]), std=float(payload["std"]))


class MoleculePool:
    """An in-memory pool of featurised molecules with docking scores.

    Attributes:
        smiles: Canonical SMILES in pool order.
        scores: Docking scores in kcal/mol, aligned with ``smiles``.
        graphs: Featurised graphs, aligned with ``smiles``.
    """

    def __init__(self, smiles: Sequence[str], scores: Sequence[float]) -> None:
        if len(smiles) != len(scores):
            raise ValueError("smiles and scores must be the same length")

        kept_smiles: list[str] = []
        kept_scores: list[float] = []
        graphs: list[Data] = []
        dropped = 0
        for smi, score in zip(smiles, scores, strict=True):
            graph = mol_to_graph(smi, y=float(score), index=len(kept_smiles))
            if graph is None:
                dropped += 1
                continue
            kept_smiles.append(smi)
            kept_scores.append(float(score))
            graphs.append(graph)

        if dropped:
            logger.warning("Dropped %d molecule(s) that could not be featurised.", dropped)
        if not graphs:
            raise ValueError("no molecules could be featurised")

        self.smiles: list[str] = kept_smiles
        self.scores: np.ndarray = np.asarray(kept_scores, dtype=np.float64)
        self.graphs: list[Data] = graphs
        logger.info("Featurised pool of %d molecules.", len(self.graphs))

    def __len__(self) -> int:
        return len(self.graphs)

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, smiles_column: str = "canonical_smiles", score_column: str = "score"
    ) -> MoleculePool:
        """Build a pool from a dataframe of docked molecules."""
        for column in (smiles_column, score_column):
            if column not in frame.columns:
                raise KeyError(f"frame has no {column!r} column; got {list(frame.columns)}")
        clean = frame.dropna(subset=[smiles_column, score_column])
        return cls(clean[smiles_column].tolist(), clean[score_column].tolist())

    def subset(self, indices: Iterable[int], scaler: TargetScaler | None = None) -> list[Data]:
        """Return graphs at ``indices``, optionally with scaled targets.

        Graphs are cloned when scaling so the pool's own objects keep their
        raw targets; the alternative, scaling in place, makes a second call
        with a different scaler silently wrong.
        """
        selected = [self.graphs[i] for i in indices]
        if scaler is None:
            return selected
        scaled: list[Data] = []
        for graph in selected:
            clone = graph.clone()
            clone.y = torch.tensor(
                scaler.transform(np.asarray([float(graph.y.item())])), dtype=torch.float32
            )
            scaled.append(clone)
        return scaled


def make_loader(
    graphs: Sequence[Data],
    cfg: TrainConfig,
    *,
    shuffle: bool,
    seed: int,
    device_type: str,
    batch_size: int | None = None,
) -> DataLoader:
    """Build a dataloader with the config's GPU-feeding parameters.

    Two overrides are applied to the configured values, both logged:

    * ``pin_memory`` is forced off on CPU, where pinning is pure overhead.
    * ``num_workers`` is forced to 0 on Windows. The graphs are already in
      memory, so workers add no overlap to win back, while the spawn start
      method re-imports this module (and therefore RDKit) in every worker.
    """
    workers = cfg.num_workers
    if workers > 0 and os.name == "nt":
        logger.debug("Windows: forcing num_workers=0 (graphs are already in memory).")
        workers = 0
    pin = cfg.pin_memory and device_type == "cuda"
    return DataLoader(
        list(graphs),
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn if workers > 0 else None,
        generator=split_generator(seed) if shuffle else None,
        drop_last=False,
    )
