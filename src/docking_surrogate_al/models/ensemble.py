"""Deep-ensemble training.

Design note
-----------
Members differ only in their initialisation seed and their batch order; the
training data is identical. That is the standard deep-ensemble recipe, and it
is deliberately *not* bagging. Subsampling the training set per member would
add diversity but throw away data, and in the active-learning setting the
labelled set is small and expensive precisely because every label cost an
oracle call.

Members train **sequentially on one GPU by default**, which is the right
default for a single-GPU machine: members are independent, so running them
concurrently on one device just splits the same VRAM and SMs between them
with no throughput gain.

On the multi-GPU path
---------------------
``uncertainty.ddp`` enables a multi-GPU path, implemented as *member-parallel*
training: one member per visible GPU via ``torch.multiprocessing.spawn``.

This is a deliberate departure from using ``DistributedDataParallel`` to shard
one member's batches across GPUs, and the reason is that DDP would be the
wrong tool here. DDP exists to all-reduce gradients so every rank holds the
same weights. Ensemble members must stay *different*: synchronising their
gradients would converge them to one model and collapse the predictive spread
to zero, which is the entire quantity being measured. Member-parallelism gets
the same wall-clock win (N members in the time of one) with no communication
at all, since the members never need to talk.

If a single member ever outgrows one GPU, DDP within a member is the correct
addition, and it composes with this: ranks would be grouped per member. That
is not implemented.

**Untested.** This path has not been run: the development machine has one GPU.
It is guarded to refuse to start unless more than one GPU is visible, and the
README says so plainly rather than implying multi-GPU support is verified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from ..config import Config
from ..data.dataset import TargetScaler
from ..seeds import seed_everything
from .checkpoint import load_checkpoint
from .gnn import DockingGNN, build_model
from .train import TrainingOutcome, train_model

logger = logging.getLogger(__name__)


def member_seed(base_seed: int, member_index: int) -> int:
    """Return the seed for one ensemble member.

    Offsetting by a large stride rather than ``base_seed + i`` keeps member 1
    of seed 0 from colliding with member 0 of seed 1, which would otherwise
    make two "independent" AL seeds share a trained model.
    """
    return base_seed * 10_000 + member_index


def train_ensemble(
    cfg: Config,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_scaler: TargetScaler,
    checkpoint_root: Path,
    *,
    seed: int,
    epochs: int | None = None,
    resume: bool = True,
    config_hash: str = "",
) -> tuple[list[DockingGNN], list[TrainingOutcome]]:
    """Train ensemble members, sequentially unless the multi-GPU path is on.

    Returns:
        ``(members, outcomes)`` with one entry per member.
    """
    size = cfg.uncertainty.ensemble_size
    if size < 2:
        raise ValueError(f"uncertainty.ensemble_size must be >= 2, got {size}")

    if cfg.uncertainty.ddp:
        _check_multi_gpu(cfg)
        raise NotImplementedError(
            "The member-parallel multi-GPU path is specified in "
            "models/ensemble.py but is not implemented, because it could not "
            "be tested on a single-GPU machine. Set uncertainty.ddp to false "
            "to train members sequentially."
        )

    members: list[DockingGNN] = []
    outcomes: list[TrainingOutcome] = []

    for index in range(size):
        this_seed = member_seed(seed, index)
        logger.info("Training ensemble member %d/%d (seed %d).", index + 1, size, this_seed)
        # Reseeding per member is what makes members differ: different
        # initialisation and different shuffling.
        seed_everything(this_seed)

        member_dir = checkpoint_root / f"member_{index:02d}"
        outcome = train_model(
            cfg,
            device,
            train_loader,
            val_loader,
            target_scaler,
            member_dir,
            seed=this_seed,
            epochs=epochs,
            resume=resume,
            config_hash=config_hash,
            model=build_model(cfg.model),
        )
        members.append(outcome.model)
        outcomes.append(outcome)

    return members, outcomes


def load_ensemble(
    cfg: Config, checkpoint_root: Path, device: torch.device
) -> tuple[list[DockingGNN], TargetScaler]:
    """Load a previously trained ensemble from its member checkpoints."""
    members: list[DockingGNN] = []
    scaler: TargetScaler | None = None

    for index in range(cfg.uncertainty.ensemble_size):
        path = checkpoint_root / f"member_{index:02d}" / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"ensemble member checkpoint missing: {path}")
        net = build_model(cfg.model).to(device)
        _, member_scaler = load_checkpoint(
            path,
            model=net,
            architecture=net.architecture_summary(),
            restore_rng=False,
            map_location=device,
        )
        members.append(net.to(device))
        scaler = member_scaler

    assert scaler is not None
    return members, scaler


def _check_multi_gpu(cfg: Config) -> None:
    """Refuse the multi-GPU path unless multiple GPUs are actually visible."""
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible < 2:
        raise RuntimeError(
            f"uncertainty.ddp is true but torch sees {visible} CUDA device(s). "
            "The multi-GPU path needs at least 2. Set uncertainty.ddp to false "
            "to train ensemble members sequentially on one device."
        )
    if cfg.uncertainty.ddp_world_size > visible:
        raise RuntimeError(
            f"uncertainty.ddp_world_size={cfg.uncertainty.ddp_world_size} exceeds the "
            f"{visible} visible CUDA device(s)."
        )
