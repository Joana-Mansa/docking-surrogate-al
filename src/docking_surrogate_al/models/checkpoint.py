"""Checkpoint saving and resuming.

Design note
-----------
A checkpoint here is not just weights: it is everything needed to continue the
run as if it had never stopped. That includes the optimiser and scheduler
state, the AMP scaler state, the early-stopping bookkeeping, the target scaler
and the RNG states for Python, NumPy and torch. Omitting the RNG states is the
usual reason a "resumed" run silently diverges from an uninterrupted one.

Writes are atomic: the payload goes to a temporary file in the same directory
and is then renamed over the target. A checkpoint written non-atomically is
worse than no checkpoint, because a kill during the write destroys the last
good one and the run resumes from nothing.

Two guards run on load, both failing loudly rather than adapting:

* ``featuriser_version`` must match, since a changed feature set silently
  mis-indexes every input column.
* The architecture summary must match, since a changed hidden width loads with
  shape errors at best and wrong semantics at worst.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..data.dataset import TargetScaler
from ..data.featurise import FEATURISER_VERSION

logger = logging.getLogger(__name__)


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint does not match the current code or config."""


@dataclass
class TrainingState:
    """Mutable bookkeeping carried across epochs and resumes."""

    epoch: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    history: list[dict[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    state: TrainingState,
    target_scaler: TargetScaler,
    architecture: dict[str, Any],
    config_hash: str,
    seed: int,
) -> None:
    """Atomically write a full checkpoint."""
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "amp_scaler_state": scaler.state_dict() if scaler is not None else None,
        "training_state": {
            "epoch": state.epoch,
            "best_val_loss": state.best_val_loss,
            "best_epoch": state.best_epoch,
            "epochs_without_improvement": state.epochs_without_improvement,
            "history": state.history,
        },
        "target_scaler": target_scaler.to_dict(),
        "architecture": architecture,
        "featuriser_version": FEATURISER_VERSION,
        "config_hash": config_hash,
        "seed": seed,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    logger.debug("Checkpoint written to %s (epoch %d).", path, state.epoch)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    architecture: dict[str, Any] | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[TrainingState, TargetScaler]:
    """Load a checkpoint into the given objects.

    Args:
        path: Checkpoint file.
        model: Model to load weights into.
        optimizer: Optimiser to restore, if resuming.
        scheduler: LR scheduler to restore, if resuming.
        scaler: AMP GradScaler to restore, if resuming.
        architecture: Current architecture summary, checked against the file.
        restore_rng: Restore RNG states so a resumed run matches an
            uninterrupted one.
        map_location: Device to map tensors onto.

    Returns:
        ``(training_state, target_scaler)``.

    Raises:
        CheckpointMismatchError: On featuriser or architecture mismatch.
    """
    # weights_only=False is required because the payload carries RNG state and
    # plain-Python bookkeeping, not just tensors. These checkpoints are
    # produced by this pipeline; do not point this at untrusted files.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    saved_featuriser = payload.get("featuriser_version")
    if saved_featuriser != FEATURISER_VERSION:
        raise CheckpointMismatchError(
            f"{path} was written with featuriser version {saved_featuriser!r} but this "
            f"code uses {FEATURISER_VERSION!r}. The feature columns have changed, so "
            "the weights cannot be reused. Retrain, or check out the matching commit."
        )

    if architecture is not None:
        saved_arch = payload.get("architecture", {})
        differing = {
            k: (saved_arch.get(k), architecture.get(k))
            for k in ("hidden_dim", "num_layers", "pool", "atom_feature_dim", "bond_feature_dim")
            if saved_arch.get(k) != architecture.get(k)
        }
        if differing:
            raise CheckpointMismatchError(
                f"{path} architecture does not match the configured model. "
                f"Differences (saved, current): {differing}"
            )

    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("amp_scaler_state") is not None:
        scaler.load_state_dict(payload["amp_scaler_state"])

    if restore_rng and payload.get("rng_state"):
        rng = payload["rng_state"]
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["torch_cuda"]])
        except (TypeError, ValueError, RuntimeError) as exc:
            # A machine with a different GPU count cannot restore CUDA RNG;
            # that is a reproducibility caveat, not a reason to refuse to run.
            logger.warning("Could not fully restore RNG state from %s: %s", path, exc)

    saved_state = payload.get("training_state", {})
    state = TrainingState(
        epoch=int(saved_state.get("epoch", 0)),
        best_val_loss=float(saved_state.get("best_val_loss", float("inf"))),
        best_epoch=int(saved_state.get("best_epoch", -1)),
        epochs_without_improvement=int(saved_state.get("epochs_without_improvement", 0)),
        history=list(saved_state.get("history", [])),
    )
    target_scaler = TargetScaler.from_dict(payload["target_scaler"])
    logger.info("Resumed from %s at epoch %d (best val %.4f).", path, state.epoch, state.best_val_loss)
    return state, target_scaler


def find_resume_checkpoint(directory: Path) -> Path | None:
    """Return the last checkpoint in ``directory``, if one exists."""
    last = directory / "last.pt"
    return last if last.exists() else None
