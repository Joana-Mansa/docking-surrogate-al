"""Predictive uncertainty: MC-dropout and deep ensembles.

Design note
-----------
Both methods return a mean and a standard deviation per molecule, in kcal/mol,
so the active-learning strategies and the calibration analysis are indifferent
to which one produced them.

MC-dropout, batched
-------------------
The naive implementation loops ``mc_passes`` times over the dataloader, which
means ``mc_passes`` kernel launches per batch and a GPU that spends most of
its time idle between tiny graphs. Instead, each batch is *replicated*
``mc_chunk`` times into a single larger graph batch and pushed through the
model in one forward pass. Dropout masks are sampled independently per replica
because dropout is applied per element, so the replicas are genuine
independent stochastic passes, not copies.

``mc_chunk`` exists because replication multiplies activation memory: the
effective batch is ``batch_size * mc_chunk`` graphs. Thirty passes with a
chunk of ten is three forward passes rather than thirty, at three times the
memory of one. On a 6 GB card that trade is right; on a larger card raise
``mc_chunk`` toward ``mc_passes``.

BatchNorm is kept in eval mode throughout (see
:meth:`DockingGNN.enable_mc_dropout`). If it were not, replication would also
change the batch statistics, and the "uncertainty" would partly measure how
many copies happened to be in the batch.

What the numbers mean
---------------------
MC-dropout approximates a posterior over weights, so its spread is *epistemic*
uncertainty: disagreement about molecules unlike the training set. It does not
capture the oracle's own noise, and it is a crude approximation. This is why
calibration is measured and reported rather than assumed; an uncalibrated
uncertainty can still rank molecules usefully for acquisition while being
quantitatively wrong, and the reliability plots show which of those is
happening.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ..data.dataset import TargetScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UncertainPrediction:
    """Per-molecule predictive mean and spread, in kcal/mol."""

    mean: np.ndarray
    std: np.ndarray
    pool_index: np.ndarray
    method: str
    n_samples: int

    def __post_init__(self) -> None:
        if not (len(self.mean) == len(self.std) == len(self.pool_index)):
            raise ValueError("mean, std and pool_index must be the same length")

    def reordered(self) -> UncertainPrediction:
        """Return predictions sorted by pool index.

        Dataloaders may return batches in any order and the AL loop indexes by
        pool position, so sorting here removes a whole class of silent
        misalignment bugs.
        """
        order = np.argsort(self.pool_index)
        return UncertainPrediction(
            mean=self.mean[order],
            std=self.std[order],
            pool_index=self.pool_index[order],
            method=self.method,
            n_samples=self.n_samples,
        )


@torch.no_grad()
def mc_dropout_predict(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: TargetScaler,
    *,
    n_passes: int = 30,
    chunk: int = 10,
    use_amp: bool = False,
) -> UncertainPrediction:
    """Predict with MC-dropout, batching stochastic passes on the device.

    Args:
        net: A :class:`DockingGNN` (needs ``enable_mc_dropout``).
        loader: Batches to predict over.
        device: Device to run on.
        scaler: Target scaler, to return kcal/mol rather than scaled units.
        n_passes: Total stochastic forward passes per molecule.
        chunk: Passes replicated into a single forward call.
        use_amp: Run the passes under autocast.

    Returns:
        :class:`UncertainPrediction` with mean and std in kcal/mol.
    """
    if not hasattr(net, "enable_mc_dropout"):
        raise TypeError("model does not support MC-dropout; expected a DockingGNN")
    if n_passes < 2:
        raise ValueError(f"mc_passes must be at least 2 to have a spread, got {n_passes}")

    net.enable_mc_dropout()
    chunk = max(1, min(chunk, n_passes))

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    indices: list[np.ndarray] = []

    for batch in loader:
        data_list = batch.to_data_list()
        n_graphs = len(data_list)
        samples = torch.empty((n_passes, n_graphs), dtype=torch.float32, device=device)

        done = 0
        while done < n_passes:
            this_chunk = min(chunk, n_passes - done)
            # One forward pass over `this_chunk` independent replicas.
            replicated = Batch.from_data_list(data_list * this_chunk).to(
                device, non_blocking=True
            )
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = net(
                    replicated.x,
                    replicated.edge_index,
                    replicated.edge_attr,
                    replicated.batch,
                )
            # from_data_list preserves order, so replica r occupies rows
            # [r * n_graphs, (r + 1) * n_graphs).
            samples[done : done + this_chunk] = out.float().view(this_chunk, n_graphs)
            done += this_chunk

        means.append(samples.mean(dim=0).cpu().numpy())
        # Unbiased sample std: n_passes is small, so the correction matters.
        stds.append(samples.std(dim=0, unbiased=True).cpu().numpy())
        indices.append(batch.pool_index.view(-1).cpu().numpy())

    return UncertainPrediction(
        mean=scaler.inverse(np.concatenate(means)),
        std=scaler.inverse_std(np.concatenate(stds)),
        pool_index=np.concatenate(indices),
        method="mc_dropout",
        n_samples=n_passes,
    ).reordered()


@torch.no_grad()
def ensemble_predict(
    members: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
    scaler: TargetScaler,
    *,
    use_amp: bool = False,
) -> UncertainPrediction:
    """Predict with a deep ensemble.

    Each member sees the identical batch sequence, so predictions align
    without any index bookkeeping beyond the pool index already carried on
    each graph.
    """
    if len(members) < 2:
        raise ValueError(f"a deep ensemble needs at least 2 members, got {len(members)}")

    per_member: list[np.ndarray] = []
    pool_index: np.ndarray | None = None

    for member in members:
        member.eval()
        predictions: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = member(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.append(out.float().cpu().numpy())
            indices.append(batch.pool_index.view(-1).cpu().numpy())
        member_predictions = np.concatenate(predictions)
        member_indices = np.concatenate(indices)
        order = np.argsort(member_indices)
        per_member.append(member_predictions[order])
        if pool_index is None:
            pool_index = member_indices[order]

    stacked = np.stack(per_member, axis=0)
    assert pool_index is not None
    return UncertainPrediction(
        mean=scaler.inverse(stacked.mean(axis=0)),
        std=scaler.inverse_std(stacked.std(axis=0, ddof=1)),
        pool_index=pool_index,
        method="deep_ensemble",
        n_samples=len(members),
    )


def predict_with_uncertainty(
    model_or_members: nn.Module | list[nn.Module],
    loader: DataLoader,
    device: torch.device,
    scaler: TargetScaler,
    *,
    method: str,
    n_passes: int = 30,
    chunk: int = 10,
    use_amp: bool = False,
) -> UncertainPrediction:
    """Dispatch to the configured uncertainty method."""
    if method == "mc_dropout":
        if isinstance(model_or_members, list):
            raise TypeError("mc_dropout expects a single model, not a list")
        return mc_dropout_predict(
            model_or_members,
            loader,
            device,
            scaler,
            n_passes=n_passes,
            chunk=chunk,
            use_amp=use_amp,
        )
    if method == "deep_ensemble":
        if not isinstance(model_or_members, list):
            raise TypeError("deep_ensemble expects a list of models")
        return ensemble_predict(model_or_members, loader, device, scaler, use_amp=use_amp)
    raise ValueError(
        f"unknown uncertainty.method {method!r}; expected 'mc_dropout' or 'deep_ensemble'"
    )
