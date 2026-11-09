"""Training loop with mixed precision and per-epoch checkpointing.

Design note
-----------
Mixed precision
---------------
``train.amp`` defaults to on. Autocast runs the forward pass in fp16 on CUDA
while keeping a fp32 master copy of the weights, and ``GradScaler`` rescales
the loss so small gradients do not flush to zero in fp16. On this workload it
is roughly a 1.5-2x speedup and halves activation memory, which is what lets a
6 GB card hold a useful batch size.

AMP is **automatically disabled on CPU**, with a log line. CPU autocast uses
bfloat16, and for the small matrix multiplies in a GNN of this size it is
slower than plain fp32 while adding numeric difference for no benefit. Keeping
the flag on in a CPU config is therefore harmless rather than a trap, which
matters because the smoke config is CPU-only.

The claim that AMP does not change results beyond seed variance is *checked*,
not asserted: ``scripts/check_amp_parity.py`` trains the same configuration
with AMP on and off across several seeds and compares the AMP-vs-FP32 gap to
the seed-to-seed spread. See the README for how to run it and what it reported.

Loss
----
Plain MSE on standardised targets. Huber was tried and rejected: docking score
outliers are mostly genuine strong binders, and down-weighting them is exactly
wrong when the downstream task is finding the top 1%.

Checkpointing
-------------
``last.pt`` is written after every epoch and ``best.pt`` whenever validation
improves. Both are written atomically. A run killed at any point resumes from
``last.pt`` with optimiser, scheduler, AMP scaler, early-stopping counters and
RNG states intact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from ..config import Config
from ..data.dataset import TargetScaler
from ..hardware import StageTimer
from .checkpoint import (
    TrainingState,
    find_resume_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .gnn import DockingGNN, build_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingOutcome:
    """What a training run produced."""

    model: DockingGNN
    state: TrainingState
    target_scaler: TargetScaler
    checkpoint_dir: Path
    amp_enabled: bool
    timing: dict[str, Any]


def amp_enabled_for(device: torch.device, configured: bool) -> bool:
    """Decide whether AMP is actually used, and say so."""
    if not configured:
        return False
    if device.type != "cuda":
        logger.info(
            "train.amp is on but the device is %s; mixed precision is disabled "
            "(CPU autocast is slower than fp32 for this model).",
            device.type,
        )
        return False
    return True


def train_model(
    cfg: Config,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_scaler: TargetScaler,
    checkpoint_dir: Path,
    *,
    seed: int,
    epochs: int | None = None,
    resume: bool = True,
    config_hash: str = "",
    model: DockingGNN | None = None,
) -> TrainingOutcome:
    """Train a model, checkpointing every epoch and resuming if possible.

    Args:
        cfg: Full run config.
        device: Resolved device.
        train_loader: Training batches with scaled targets.
        val_loader: Validation batches with scaled targets.
        target_scaler: Scaler fitted on the training partition.
        checkpoint_dir: Directory for ``last.pt`` and ``best.pt``.
        seed: Seed to record with the result.
        epochs: Override ``cfg.train.epochs`` (used by AL rounds).
        resume: Resume from ``last.pt`` if present.
        config_hash: Recorded in the checkpoint.
        model: Reuse an existing model instance instead of building one.

    Returns:
        A :class:`TrainingOutcome` holding the best-validation weights.
    """
    total_epochs = epochs if epochs is not None else cfg.train.epochs
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    net = model if model is not None else build_model(cfg.model)
    net = net.to(device)
    architecture = net.architecture_summary()

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.train.lr_scheduler_patience
    )

    use_amp = amp_enabled_for(device, cfg.train.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.MSELoss()

    state = TrainingState()
    resume_path = find_resume_checkpoint(checkpoint_dir) if resume else None
    if resume_path is not None:
        state, target_scaler = load_checkpoint(
            resume_path,
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            architecture=architecture,
            map_location=device,
        )
        net = net.to(device)

    if state.epoch >= total_epochs:
        logger.info(
            "Checkpoint is already at epoch %d of %d; nothing to train.",
            state.epoch,
            total_epochs,
        )

    with StageTimer("train", device) as timer:
        for epoch in range(state.epoch + 1, total_epochs + 1):
            # Shuffling is seeded per epoch, not once per loader. The loader's
            # generator is recreated on resume, so without this epoch 4 of a
            # resumed run would shuffle like epoch 1 of a fresh one and the
            # resumed run would silently diverge from an uninterrupted one.
            _seed_epoch_shuffle(train_loader, seed, epoch)
            train_loss, n_seen = _train_one_epoch(
                net, train_loader, optimizer, criterion, scaler, device, use_amp, cfg.train.grad_clip
            )
            val_loss = _evaluate_loss(net, val_loader, criterion, device, use_amp)
            scheduler.step(val_loss)
            timer.add_samples(n_seen)

            state.epoch = epoch
            state.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            improved = val_loss < state.best_val_loss - 1e-6
            if improved:
                state.best_val_loss = val_loss
                state.best_epoch = epoch
                state.epochs_without_improvement = 0
            else:
                state.epochs_without_improvement += 1

            if epoch % 10 == 0 or improved or epoch == total_epochs:
                logger.info(
                    "epoch %3d/%d  train %.4f  val %.4f  lr %.2e%s",
                    epoch,
                    total_epochs,
                    train_loss,
                    val_loss,
                    optimizer.param_groups[0]["lr"],
                    "  *" if improved else "",
                )

            save_kwargs = {
                "model": net,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "scaler": scaler if use_amp else None,
                "state": state,
                "target_scaler": target_scaler,
                "architecture": architecture,
                "config_hash": config_hash,
                "seed": seed,
            }
            if cfg.train.checkpoint_every_epoch:
                save_checkpoint(checkpoint_dir / "last.pt", **save_kwargs)
            if improved:
                save_checkpoint(checkpoint_dir / "best.pt", **save_kwargs)

            if state.epochs_without_improvement >= cfg.train.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d: no improvement for %d epochs "
                    "(best %.4f at epoch %d).",
                    epoch,
                    state.epochs_without_improvement,
                    state.best_val_loss,
                    state.best_epoch,
                )
                break

    # Report the best weights, not the last ones.
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        load_checkpoint(
            best_path, model=net, architecture=architecture, restore_rng=False, map_location=device
        )
        net = net.to(device)

    return TrainingOutcome(
        model=net,
        state=state,
        target_scaler=target_scaler,
        checkpoint_dir=checkpoint_dir,
        amp_enabled=use_amp,
        timing=timer.summary(),
    )


def _seed_epoch_shuffle(loader: DataLoader, seed: int, epoch: int) -> None:
    """Reseed a shuffled loader's generator deterministically for this epoch.

    Makes the batch order a pure function of ``(seed, epoch)`` rather than of
    how many epochs this particular loader object has already served, which is
    what lets a resumed run reproduce an uninterrupted one exactly.
    """
    generator = getattr(loader, "generator", None)
    if generator is None:
        return
    # Mixed into a 63-bit value so distinct (seed, epoch) pairs stay distinct.
    generator.manual_seed((seed * 1_000_003 + epoch) % (2**63 - 1))


def _train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> tuple[float, int]:
    """Run one training epoch; return (mean loss, graphs seen)."""
    net.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            predictions = net(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(predictions, batch.y.view(-1))

        scaler.scale(loss).backward()
        if grad_clip > 0:
            # Unscale before clipping, or the clip threshold is applied to
            # scaled gradients and effectively does nothing.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        n_graphs = int(batch.num_graphs)
        total_loss += float(loss.detach()) * n_graphs
        total_graphs += n_graphs

    return total_loss / max(total_graphs, 1), total_graphs


@torch.no_grad()
def _evaluate_loss(
    net: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Mean validation loss in scaled target space."""
    net.eval()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            predictions = net(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(predictions, batch.y.view(-1))
        n_graphs = int(batch.num_graphs)
        total_loss += float(loss.detach()) * n_graphs
        total_graphs += n_graphs
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def predict_mean(
    net: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic predictions in scaled space, with pool indices.

    Returns:
        ``(predictions, pool_indices)``. Pool indices are carried through the
        graphs so predictions can be realigned after shuffling.
    """
    net.eval()
    predictions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = net(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        predictions.append(out.float().cpu().numpy())
        indices.append(batch.pool_id.view(-1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(indices)
