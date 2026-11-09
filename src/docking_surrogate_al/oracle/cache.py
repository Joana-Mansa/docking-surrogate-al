"""Interruption-safe docking cache.

Design note
-----------
Docking is the expensive part of this project: a 5000-molecule pool at a few
seconds per molecule is hours of CPU time, and at 50000 it is days. Losing
that work to a crash, a full disk or a closed laptop lid is unacceptable, so
the cache is an append-only JSONL file written *incrementally*:

* one JSON object per line, appended as each molecule finishes
* ``flush()`` plus ``os.fsync()`` after every write, so a power loss costs at
  most the record in flight rather than the whole stage
* a truncated final line (the signature of a kill mid-write) is skipped on
  load with a warning, not treated as corruption of the whole file

Append-only also means the file is a log, not a database: re-docking a
molecule appends a second record and the last one wins. That keeps writes
atomic without a lock, which matters because docking is parallelised over
processes. Only the parent process writes; workers return results.

Keying
------
The key is ``(canonical_smiles, settings_fingerprint)``. SMILES alone is not
enough: the same molecule docked against a different box, or with different
exhaustiveness, or by a different backend, is a different number. Partitioning
by fingerprint means changing the box starts a fresh set of entries instead of
silently reusing stale ones, while a rerun with unchanged settings reuses
everything. Each record stores the backend that produced it, so provenance
survives even if two backends share a cache file.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator

from .base import OracleResult

logger = logging.getLogger(__name__)


class DockingCache:
    """Append-only JSONL cache of docking results."""

    def __init__(self, path: Path, settings_fingerprint: str) -> None:
        self.path = Path(path)
        self.fingerprint = settings_fingerprint
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, OracleResult] = {}
        self._handle = None
        self._load()

    # -- loading ----------------------------------------------------------
    def _load(self) -> None:
        """Read existing records for this fingerprint."""
        if not self.path.exists():
            logger.info("Docking cache %s does not exist yet; starting empty.", self.path)
            return

        kept = 0
        skipped_fingerprint = 0
        malformed = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Almost always a half-written final line from a hard kill.
                    malformed += 1
                    logger.warning(
                        "Skipping malformed cache line %d in %s (likely an "
                        "interrupted write).",
                        line_number,
                        self.path,
                    )
                    continue
                if record.get("settings_fingerprint") != self.fingerprint:
                    skipped_fingerprint += 1
                    continue
                result = OracleResult(
                    smiles=record["smiles"],
                    score=record.get("score"),
                    backend=record.get("backend", "unknown"),
                    backend_version=record.get("backend_version", "unknown"),
                    status=record.get("status", "ok"),
                    error=record.get("error"),
                    wall_clock_s=record.get("wall_clock_s"),
                    extra=record.get("extra", {}),
                )
                # Last record for a SMILES wins; the file is a log.
                self._entries[result.smiles] = result
                kept += 1

        logger.info(
            "Loaded %d cache record(s) for fingerprint %s from %s "
            "(%d for other settings, %d malformed).",
            kept,
            self.fingerprint[:12],
            self.path,
            skipped_fingerprint,
            malformed,
        )

    # -- reading ----------------------------------------------------------
    def __contains__(self, smiles: str) -> bool:
        return smiles in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, smiles: str) -> OracleResult | None:
        """Return the cached result for ``smiles``, if any."""
        return self._entries.get(smiles)

    def successful(self) -> dict[str, float]:
        """Return ``{smiles: score}`` for every successfully docked molecule."""
        return {s: r.score for s, r in self._entries.items() if r.ok and r.score is not None}

    def items(self) -> Iterator[tuple[str, OracleResult]]:
        """Iterate over all cached results."""
        return iter(self._entries.items())

    def missing(self, smiles_list: list[str], retry_failures: bool = False) -> list[str]:
        """Return the molecules that still need docking.

        Args:
            smiles_list: Candidate molecules.
            retry_failures: If True, molecules whose cached result is a
                failure are returned for another attempt. Off by default: a
                molecule that failed conformer generation will fail again,
                and retrying it every run wastes the time the cache saves.
        """
        todo: list[str] = []
        for smiles in smiles_list:
            cached = self._entries.get(smiles)
            if cached is None or (retry_failures and not cached.ok):
                todo.append(smiles)
        return todo

    # -- writing ----------------------------------------------------------
    def append(self, result: OracleResult) -> None:
        """Append one result and fsync it to disk immediately."""
        record = result.to_dict()
        record["settings_fingerprint"] = self.fingerprint
        handle = self._open()
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._entries[result.smiles] = result

    def _open(self):  # noqa: ANN202 - returns a TextIO, kept private
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    def close(self) -> None:
        """Close the append handle."""
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def __enter__(self) -> DockingCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reporting --------------------------------------------------------
    def summary(self) -> dict[str, object]:
        """Return a manifest-ready summary, including backend provenance."""
        statuses: dict[str, int] = {}
        backends: dict[str, int] = {}
        for result in self._entries.values():
            statuses[result.status] = statuses.get(result.status, 0) + 1
            label = f"{result.backend}:{result.backend_version}"
            backends[label] = backends.get(label, 0) + 1
        return {
            "cache_path": str(self.path),
            "settings_fingerprint": self.fingerprint,
            "total_records": len(self._entries),
            "status_counts": statuses,
            "backend_counts": backends,
        }
