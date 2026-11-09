"""Run manifests.

Design note
-----------
A CSV of numbers is not a result; a CSV plus the exact conditions that
produced it is. Every stage writes into a single JSON manifest per run
recording:

* the resolved config and its SHA-256 hash
* the git commit and whether the working tree was dirty at launch
* installed versions of the packages that can change numeric output
* hardware (see :mod:`docking_surrogate_al.hardware`)
* the seed, and wall-clock time, throughput and peak VRAM per stage

The manifest is rewritten after every stage rather than only at the end, so an
interrupted run still documents what it completed. Writes are atomic (write to
a temporary file in the same directory, then replace) so a crash mid-write
cannot leave an unparseable manifest behind.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Config, config_hash
from .hardware import HardwareInfo

# Packages whose version can plausibly change a reported number.
TRACKED_PACKAGES = (
    "torch",
    "torch-geometric",
    "rdkit",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _git_info() -> dict[str, Any]:
    """Return the current commit and working-tree cleanliness.

    Absence of git, or a non-repository checkout (a tarball download, say), is
    recorded rather than raised: the manifest should still be written.
    """

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "commit": commit,
        "branch": branch,
        # None means "unknown", which is different from "clean".
        "dirty": None if status is None else bool(status),
    }


class RunManifest:
    """Incrementally written JSON record of one run."""

    def __init__(self, cfg: Config, hardware: HardwareInfo, stage_group: str) -> None:
        self.path = cfg.run_dir / "manifest.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load_existing()
        self._data.setdefault("run_name", cfg.run.name)
        self._data.setdefault("created_utc", _now())
        self._data["updated_utc"] = _now()
        self._data["seed"] = cfg.run.seed
        self._data["config_hash"] = config_hash(cfg)
        self._data["config"] = cfg.to_dict()
        self._data["git"] = _git_info()
        self._data["packages"] = _package_versions()
        self._data["hardware"] = hardware.to_dict()
        self._data["env"] = {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        self._data.setdefault("stages", {})
        self._data["last_stage_group"] = stage_group
        self.flush()

    def _load_existing(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            # A truncated manifest from a hard kill should not block a rerun.
            return {}

    def record_stage(self, summary: dict[str, Any]) -> None:
        """Record one stage summary (see :meth:`StageTimer.summary`)."""
        stage = summary["stage"]
        self._data["stages"][stage] = {**summary, "finished_utc": _now()}
        self.flush()

    def record(self, key: str, value: Any) -> None:
        """Record an arbitrary top-level key, e.g. dataset sizes."""
        self._data[key] = value
        self.flush()

    def flush(self) -> None:
        """Atomically write the manifest to disk."""
        self._data["updated_utc"] = _now()
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(self.path)

    @property
    def data(self) -> dict[str, Any]:
        """The manifest contents."""
        return self._data


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
