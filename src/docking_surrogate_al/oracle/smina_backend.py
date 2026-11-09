"""smina docking backend.

Design note
-----------
smina is CPU-only. Parallelism therefore comes from running many *processes*
over CPU cores (see :mod:`docking_surrogate_al.oracle.runner`), and each smina
invocation is pinned to a single CPU thread with ``--cpu 1``. Letting smina
use its own internal threading as well would oversubscribe the machine: N
workers each spawning N threads is N^2 threads fighting for N cores, which in
practice is slower than one thread per worker.

Scores are read from smina's affinity table on stdout. The best (most
negative) mode is the surrogate's regression target; all modes are kept in
``extra`` so that pose-spread analyses remain possible without re-docking.

Determinism: smina's search is stochastic, so ``--seed`` is passed explicitly
from the config. With a fixed seed, exhaustiveness and box, re-docking a
molecule reproduces its score. Different exhaustiveness gives different
numbers, which is exactly why exhaustiveness is part of the cache fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

from ..config import OracleConfig, TargetConfig, resolve_path
from .base import DockingOracle, OracleResult, OracleUnavailableError
from .prepare import LigandPrepError, clean_receptor_pdb, convert_to_pdbqt, generate_conformer_sdf

logger = logging.getLogger(__name__)

# Matches a row of smina's affinity table, e.g. "   1      -7.12   0.000  0.000".
_MODE_ROW = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


class SminaOracle(DockingOracle):
    """Dock molecules against one prepared target with smina."""

    name = "smina"

    def __init__(self, oracle_cfg: OracleConfig, target_cfg: TargetConfig) -> None:
        self.cfg = oracle_cfg
        self.target = target_cfg
        self._version: str | None = None
        self._receptor_path: Path | None = None

    # -- setup ------------------------------------------------------------
    def _binary(self) -> str:
        binary = shutil.which(self.cfg.smina_binary)
        if binary is None:
            raise OracleUnavailableError(
                f"smina binary {self.cfg.smina_binary!r} was not found on PATH.\n"
                "Install it with one of:\n"
                "  conda install -c conda-forge smina\n"
                "  apt-get install smina            (Debian/Ubuntu)\n"
                "or download a static build from https://sourceforge.net/projects/smina/ "
                "and set oracle.smina_binary to its full path."
            )
        return binary

    def version(self) -> str:
        """Return smina's reported version, querying the binary once."""
        if self._version is not None:
            return self._version
        try:
            result = subprocess.run(
                [self._binary(), "--version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            first_line = (result.stdout or result.stderr).strip().splitlines()
            self._version = first_line[0].strip() if first_line else "unknown"
        except (OSError, subprocess.SubprocessError) as exc:
            raise OracleUnavailableError(f"could not run smina --version: {exc}") from exc
        return self._version

    def settings_fingerprint(self) -> str:
        """Hash every setting that can change a docking score."""
        payload = {
            "backend": self.name,
            "backend_version": self.version(),
            "target_name": self.target.name,
            "receptor_pdb": self.target.receptor_pdb,
            "center": list(self.target.center),
            "size": list(self.target.size),
            "exhaustiveness": self.cfg.exhaustiveness,
            "num_modes": self.cfg.num_modes,
            "seed": self.cfg.seed,
            "embed_max_attempts": self.cfg.embed_max_attempts,
            "ff_max_iters": self.cfg.ff_max_iters,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def prepare(self) -> dict[str, object]:
        """Prepare the receptor once, before any docking.

        Returns:
            Preparation statistics for the run manifest.
        """
        self._binary()  # fail fast if smina is missing
        raw = resolve_path(self.target.receptor_pdb)
        pdbqt = resolve_path(self.target.receptor_pdbqt)
        cleaned = pdbqt.with_name(pdbqt.stem + "_clean.pdb")

        stats = clean_receptor_pdb(raw, cleaned)
        if convert_to_pdbqt(cleaned, pdbqt):
            self._receptor_path = pdbqt
        else:
            self._receptor_path = cleaned
        stats["receptor_used"] = str(self._receptor_path)
        stats["smina_version"] = self.version()
        return stats

    def _receptor(self) -> Path:
        if self._receptor_path is None:
            # Worker processes receive the oracle after prepare() ran in the
            # parent; resolve the expected artefact rather than re-preparing.
            pdbqt = resolve_path(self.target.receptor_pdbqt)
            cleaned = pdbqt.with_name(pdbqt.stem + "_clean.pdb")
            if pdbqt.exists():
                self._receptor_path = pdbqt
            elif cleaned.exists():
                self._receptor_path = cleaned
            else:
                raise OracleUnavailableError(
                    f"no prepared receptor at {pdbqt} or {cleaned}; call prepare() first"
                )
        return self._receptor_path

    # -- docking ----------------------------------------------------------
    def dock(self, smiles: str) -> OracleResult:
        """Dock one molecule, returning a failure result rather than raising."""
        start = perf_counter()
        try:
            version = self.version()
        except OracleUnavailableError:
            raise

        with tempfile.TemporaryDirectory(prefix="dsal_dock_") as tmp:
            tmp_dir = Path(tmp)
            ligand = tmp_dir / "ligand.sdf"
            try:
                generate_conformer_sdf(
                    smiles,
                    ligand,
                    seed=self.cfg.seed,
                    max_attempts=self.cfg.embed_max_attempts,
                    ff_max_iters=self.cfg.ff_max_iters,
                )
            except LigandPrepError as exc:
                return OracleResult(
                    smiles=smiles,
                    score=None,
                    backend=self.name,
                    backend_version=version,
                    status="prep_failed",
                    error=str(exc),
                    wall_clock_s=round(perf_counter() - start, 3),
                )

            command = [
                self._binary(),
                "--receptor", str(self._receptor()),
                "--ligand", str(ligand),
                "--center_x", str(self.target.center[0]),
                "--center_y", str(self.target.center[1]),
                "--center_z", str(self.target.center[2]),
                "--size_x", str(self.target.size[0]),
                "--size_y", str(self.target.size[1]),
                "--size_z", str(self.target.size[2]),
                "--exhaustiveness", str(self.cfg.exhaustiveness),
                "--num_modes", str(self.cfg.num_modes),
                "--seed", str(self.cfg.seed),
                # One thread per process; the runner provides the parallelism.
                "--cpu", "1",
                "--out", str(tmp_dir / "out.sdf"),
            ]

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return OracleResult(
                    smiles=smiles,
                    score=None,
                    backend=self.name,
                    backend_version=version,
                    status="timeout",
                    error=f"smina exceeded {self.cfg.timeout_s}s",
                    wall_clock_s=round(perf_counter() - start, 3),
                )
            except OSError as exc:
                return OracleResult(
                    smiles=smiles,
                    score=None,
                    backend=self.name,
                    backend_version=version,
                    status="dock_failed",
                    error=str(exc),
                    wall_clock_s=round(perf_counter() - start, 3),
                )

            if result.returncode != 0:
                return OracleResult(
                    smiles=smiles,
                    score=None,
                    backend=self.name,
                    backend_version=version,
                    status="dock_failed",
                    error=f"smina exit {result.returncode}: {result.stderr.strip()[:400]}",
                    wall_clock_s=round(perf_counter() - start, 3),
                )

            modes = parse_affinity_table(result.stdout)
            if not modes:
                return OracleResult(
                    smiles=smiles,
                    score=None,
                    backend=self.name,
                    backend_version=version,
                    status="parse_failed",
                    error=f"no affinity table in smina output: {result.stdout.strip()[:400]}",
                    wall_clock_s=round(perf_counter() - start, 3),
                )

            return OracleResult(
                smiles=smiles,
                score=min(modes),
                backend=self.name,
                backend_version=version,
                status="ok",
                wall_clock_s=round(perf_counter() - start, 3),
                extra={"all_mode_affinities": modes},
            )


def parse_affinity_table(stdout: str) -> list[float]:
    """Extract per-mode affinities (kcal/mol) from smina's stdout.

    Kept as a module-level function with no smina dependency so it can be
    unit-tested against captured output without a binary installed.
    """
    affinities: list[float] = []
    for line in stdout.splitlines():
        match = _MODE_ROW.match(line)
        if match:
            affinities.append(float(match.group(2)))
    return affinities
