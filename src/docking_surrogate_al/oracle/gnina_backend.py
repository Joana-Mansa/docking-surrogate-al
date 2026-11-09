"""gnina backend placeholder.

Design note
-----------
gnina is a CUDA-capable docking tool: it performs the same Monte Carlo search
as smina, then rescores the resulting poses with a convolutional neural
network. Adding it is the reason the oracle is an interface at all.

This module is deliberately a documented stub rather than an untested
implementation. Writing speculative subprocess plumbing for a binary that was
never run would be worse than nothing: it would look finished, and the first
person to enable it would debug someone else's guesses. What is real here is
the *contract* it has to satisfy, so that filling it in touches nothing
outside this file.

To implement it:

1. Mirror :class:`~docking_surrogate_al.oracle.smina_backend.SminaOracle`. The
   command line is nearly identical; add ``--cnn_scoring rescore`` (or
   ``refinement``) and keep ``--seed``.
2. Report a *different* ``name`` (``"gnina"``). The cache stores the backend
   with every score, and the settings fingerprint includes the backend name
   and version, so gnina scores will land in their own cache partition
   automatically and can never be silently mixed with smina scores.
3. Decide which number is the target. gnina reports both an affinity and a
   CNN score, and they are not interchangeable. Put the choice in the config
   and record it in the fingerprint.
4. GPU contention is the one genuinely new concern: unlike smina, gnina uses
   the GPU, so the CPU-core fan-out in
   :mod:`docking_surrogate_al.oracle.runner` is the wrong parallelism model
   for it. Limit concurrency to the number of GPUs, or serialise.
"""

from __future__ import annotations

from typing import Any

from ..config import OracleConfig, TargetConfig
from .base import DockingOracle, OracleResult


class GninaOracle(DockingOracle):
    """Not implemented; see the module docstring for the contract."""

    name = "gnina"

    def __init__(self, oracle_cfg: OracleConfig, target_cfg: TargetConfig) -> None:
        self.cfg = oracle_cfg
        self.target = target_cfg
        raise NotImplementedError(
            "The gnina backend is a documented stub, not a working backend. "
            "Set oracle.backend to 'smina'. See "
            "src/docking_surrogate_al/oracle/gnina_backend.py for what "
            "implementing it requires."
        )

    def version(self) -> str:
        """Unimplemented."""
        raise NotImplementedError

    def settings_fingerprint(self) -> str:
        """Unimplemented."""
        raise NotImplementedError

    def prepare(self) -> dict[str, Any]:
        """Unimplemented."""
        raise NotImplementedError

    def dock(self, smiles: str) -> OracleResult:
        """Unimplemented."""
        raise NotImplementedError
