"""Deterministic seeding.

Design note
-----------
Every result this project reports is tagged with the seed that produced it.
``seed_everything`` covers Python, NumPy and torch (CPU and all CUDA devices).

Determinism has a cost and a limit, both of which matter for honest reporting:

* ``deterministic=True`` sets ``torch.use_deterministic_algorithms`` and the
  cuBLAS workspace variable. Some scatter/gather kernels used by message
  passing have no deterministic CUDA implementation, so this can raise. We
  therefore default to ``warn_only=True``: the run stays as reproducible as
  the kernels allow, and prints which operations were not.
* Even fully seeded, results vary across GPU models, driver versions and
  library versions. Seed variance bands in the figures are computed across
  *seeds on one machine*; they do not bound cross-machine variation.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch.

    Args:
        seed: The seed to apply.
        deterministic: Request deterministic kernels where they exist.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Must be set before the first cuBLAS handle is created to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Seeded everything with seed=%d (deterministic=%s).", seed, deterministic)


def worker_init_fn(worker_id: int) -> None:
    """Seed each dataloader worker distinctly but reproducibly."""
    base = torch.initial_seed() % 2**32
    seed = (base + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def split_generator(seed: int) -> torch.Generator:
    """Return a torch generator for dataloader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
