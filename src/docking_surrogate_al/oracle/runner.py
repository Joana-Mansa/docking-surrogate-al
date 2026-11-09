"""Parallel docking over CPU cores.

Design note
-----------
smina is CPU-only, so this is a *process* pool over CPU cores and has nothing
to do with the GPU settings elsewhere in the config. Processes rather than
threads, because each docking run is an external binary plus RDKit conformer
generation, both of which are CPU-bound native code.

Two properties matter more than raw speed:

* **Results are written as they arrive.** The pool yields completed molecules
  via ``imap_unordered`` and the parent appends each to the cache and fsyncs
  it immediately. An interrupted run therefore keeps every molecule that
  finished. Only the parent writes, which is what makes the append-only cache
  safe without a lock.
* **One worker cannot end the run.** Backends return failure results instead
  of raising, and the runner additionally catches anything unexpected from a
  worker so a single pathological molecule cannot take down the pool.

Default worker count is ``cpu_count() - 1``, leaving one core for the parent
process and the OS. On a laptop, docking all cores at 100% also thermally
throttles the GPU that the training stage needs.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

from .base import DockingOracle, OracleResult
from .cache import DockingCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DockingRunStats:
    """Outcome of one docking stage."""

    requested: int
    already_cached: int
    docked: int
    succeeded: int
    failed: int
    wall_clock_s: float
    workers: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a manifest-ready mapping."""
        return {
            "requested": self.requested,
            "already_cached": self.already_cached,
            "docked": self.docked,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "workers": self.workers,
            "molecules_per_second": (
                round(self.docked / self.wall_clock_s, 3) if self.wall_clock_s > 0 and self.docked else 0.0
            ),
        }


def resolve_workers(configured: int | None) -> int:
    """Resolve the CPU worker count, leaving one core for the parent."""
    if configured is not None and configured > 0:
        return configured
    return max(1, (os.cpu_count() or 2) - 1)


# The oracle is passed to each worker once at pool construction rather than
# pickled with every task, which matters when the task payload is a SMILES
# string and the oracle holds target paths and a cached version string.
_WORKER_ORACLE: DockingOracle | None = None


def _init_worker(oracle: DockingOracle) -> None:
    global _WORKER_ORACLE  # noqa: PLW0603 - process-local, set once per worker
    _WORKER_ORACLE = oracle


def _dock_one(smiles: str) -> OracleResult:
    assert _WORKER_ORACLE is not None, "worker initialised without an oracle"
    try:
        return _WORKER_ORACLE.dock(smiles)
    except Exception as exc:  # noqa: BLE001 - a worker must never kill the pool
        return OracleResult(
            smiles=smiles,
            score=None,
            backend=_WORKER_ORACLE.name,
            backend_version="unknown",
            status="dock_failed",
            error=f"unhandled worker exception: {type(exc).__name__}: {exc}",
        )


def dock_molecules(
    oracle: DockingOracle,
    smiles_list: Sequence[str],
    cache: DockingCache,
    *,
    n_workers: int | None = None,
    retry_failures: bool = False,
    progress_every: int = 25,
    on_result: Callable[[OracleResult], None] | None = None,
) -> DockingRunStats:
    """Dock every molecule not already cached, writing results incrementally.

    Args:
        oracle: The docking backend.
        smiles_list: Canonical SMILES to dock.
        cache: Cache to read from and append to.
        n_workers: CPU processes; None means ``cpu_count() - 1``.
        retry_failures: Re-attempt molecules whose cached result failed.
        progress_every: Log progress every N molecules.
        on_result: Optional callback invoked for each completed molecule.

    Returns:
        :class:`DockingRunStats` for the run manifest.
    """
    unique = list(dict.fromkeys(smiles_list))
    todo = cache.missing(unique, retry_failures=retry_failures)
    already = len(unique) - len(todo)
    workers = resolve_workers(n_workers)

    logger.info(
        "Docking stage: %d molecule(s) requested, %d already cached, %d to dock "
        "on %d CPU worker(s).",
        len(unique),
        already,
        len(todo),
        workers,
    )

    start = perf_counter()
    succeeded = 0
    failed = 0

    if todo:
        # "spawn" is the only start method available on Windows and avoids
        # fork-safety problems with RDKit's global state elsewhere.
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers, initializer=_init_worker, initargs=(oracle,)
        ) as pool:
            for finished, result in enumerate(
                pool.imap_unordered(_dock_one, todo, chunksize=1), start=1
            ):
                cache.append(result)
                if result.ok:
                    succeeded += 1
                else:
                    failed += 1
                if on_result is not None:
                    on_result(result)
                if finished % progress_every == 0 or finished == len(todo):
                    elapsed = perf_counter() - start
                    rate = finished / elapsed if elapsed > 0 else 0.0
                    remaining = (len(todo) - finished) / rate if rate > 0 else float("nan")
                    logger.info(
                        "Docked %d/%d (%d ok, %d failed) at %.2f mol/s; ~%.0fs remaining.",
                        finished,
                        len(todo),
                        succeeded,
                        failed,
                        rate,
                        remaining,
                    )

    elapsed = perf_counter() - start
    stats = DockingRunStats(
        requested=len(unique),
        already_cached=already,
        docked=len(todo),
        succeeded=succeeded,
        failed=failed,
        wall_clock_s=elapsed,
        workers=workers,
    )
    if failed:
        logger.warning(
            "%d molecule(s) failed to dock; their failure reasons are in the cache.", failed
        )
    return stats
