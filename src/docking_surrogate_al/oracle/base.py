"""Docking oracle interface.

Design note
-----------
The oracle is behind an interface for one concrete reason: smina is CPU-only,
and a CUDA-capable scorer (gnina, which rescores poses with a CNN) should be
addable without the data, model, evaluation or active-learning code noticing.
Everything upstream of this module therefore talks only to
:class:`DockingOracle`, and nothing upstream imports ``smina`` directly.

Two consequences shape the design:

* **Provenance travels with the number.** Every :class:`OracleResult` carries
  the backend name and version that produced it, and the cache stores them.
  Mixing smina and gnina scores in one dataset without being able to tell
  which is which would make the whole experiment uninterpretable, so a score
  is never stored without its producer.
* **Failure is a normal outcome.** Conformer generation fails, docking times
  out, and a fraction of any real library simply will not dock. These return
  a result with ``status != "ok"`` and ``score=None`` rather than raising, so
  one bad molecule cannot end a 5000-molecule run. Failures are counted and
  reported, never silently dropped.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

OracleStatus = Literal["ok", "prep_failed", "dock_failed", "timeout", "parse_failed"]


@dataclass(frozen=True)
class OracleResult:
    """One docking outcome.

    Attributes:
        smiles: The canonical SMILES that was docked (the cache key).
        score: Best docking score in kcal/mol, or None if docking failed.
            More negative is better, following the smina convention.
        backend: Which oracle produced this, e.g. ``"smina"``.
        backend_version: Version string reported by the backend binary.
        status: ``"ok"`` or the stage that failed.
        error: Diagnostic message when ``status != "ok"``.
        wall_clock_s: Time spent on this molecule.
        extra: Backend-specific fields, e.g. per-mode scores.
    """

    smiles: str
    score: float | None
    backend: str
    backend_version: str
    status: OracleStatus
    error: str | None = None
    wall_clock_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when a usable score was produced."""
        return self.status == "ok" and self.score is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


class DockingOracle(abc.ABC):
    """Abstract docking oracle over canonical SMILES."""

    #: Short backend identifier stored with every cached score.
    name: str

    @abc.abstractmethod
    def version(self) -> str:
        """Return the backend version string.

        Implementations should invoke the actual binary rather than returning
        a hardcoded constant, since the binary is what determines the numbers.
        """

    @abc.abstractmethod
    def settings_fingerprint(self) -> str:
        """Return a hash of every setting that can change a score.

        Target, box centre and size, exhaustiveness and backend version all
        belong here. The cache is partitioned by this fingerprint, so changing
        the box does not silently reuse scores computed against the old one.
        """

    @abc.abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Do any one-off setup, e.g. receptor preparation.

        Called once before docking begins, in the parent process.

        Returns:
            Preparation statistics for the run manifest.
        """

    @abc.abstractmethod
    def dock(self, smiles: str) -> OracleResult:
        """Dock one molecule and return its result.

        Must not raise for ordinary chemistry or tooling failures; return a
        result with the appropriate ``status`` instead.
        """


class OracleUnavailableError(RuntimeError):
    """Raised when a backend's binary or inputs are missing.

    This *is* raised rather than returned, because a missing smina binary is a
    setup error affecting every molecule, not a per-molecule failure.
    """
