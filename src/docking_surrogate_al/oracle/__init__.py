"""Docking oracle package.

Upstream code builds an oracle through :func:`make_oracle` and then talks only
to the :class:`~docking_surrogate_al.oracle.base.DockingOracle` interface, so
swapping smina for a CUDA-capable backend touches this package only.
"""

from __future__ import annotations

from ..config import OracleConfig, TargetConfig
from .base import DockingOracle, OracleResult, OracleStatus, OracleUnavailableError
from .cache import DockingCache
from .runner import DockingRunStats, dock_molecules, resolve_workers

__all__ = [
    "DockingCache",
    "DockingOracle",
    "DockingRunStats",
    "OracleResult",
    "OracleStatus",
    "OracleUnavailableError",
    "dock_molecules",
    "make_oracle",
    "resolve_workers",
]


def make_oracle(oracle_cfg: OracleConfig, target_cfg: TargetConfig) -> DockingOracle:
    """Construct the configured docking backend.

    Backends are imported lazily so that selecting smina does not require
    gnina's dependencies, and so a stub backend's import-time guard fires only
    when it is actually requested.
    """
    backend = oracle_cfg.backend.strip().lower()
    if backend == "smina":
        from .smina_backend import SminaOracle

        return SminaOracle(oracle_cfg, target_cfg)
    if backend == "gnina":
        from .gnina_backend import GninaOracle

        return GninaOracle(oracle_cfg, target_cfg)
    raise ValueError(f"unknown oracle.backend {oracle_cfg.backend!r}; expected 'smina' or 'gnina'")
