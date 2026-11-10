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
from typing import Sequence

from rdkit import Chem

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
        """Dock one molecule, returning a failure result rather than raising.

        Delegates to the same batch implementation as :meth:`dock_many` with
        a batch of one. This is not just code reuse: when the two paths
        parsed scores from different places they disagreed, because smina's
        stdout affinity table is rounded to one decimal while the output SDF
        carries full precision. The docking score is this project's
        regression target, so quantising it to 0.1 kcal/mol would coarsen
        the very thing the surrogate learns.
        """
        return self._dock_impl([smiles])[0]

    def dock_many(self, smiles_list: Sequence[str]) -> list[OracleResult]:
        """Dock a batch of molecules with a single smina invocation.

        Never lets one bad ligand cost the whole batch: ligands that cannot
        be embedded are reported individually and omitted from the SDF handed
        to smina.
        """
        return self._dock_impl(list(smiles_list))

    def _dock_impl(self, smiles_list: list[str]) -> list[OracleResult]:
        """Dock one batch; shared by dock() and dock_many()."""
        if not smiles_list:
            return []

        start = perf_counter()
        version = self.version()
        results: dict[str, OracleResult] = {}
        tag_to_smiles: dict[str, str] = {}

        with tempfile.TemporaryDirectory(prefix="dsal_batch_") as tmp:
            tmp_dir = Path(tmp)
            batch_sdf = tmp_dir / "ligands.sdf"
            out_sdf = tmp_dir / "out.sdf"

            writer = Chem.SDWriter(str(batch_sdf))
            try:
                for index, smiles in enumerate(smiles_list):
                    tag = _LIGAND_TAG.format(index=index)
                    single = tmp_dir / f"lig_{index}.sdf"
                    try:
                        generate_conformer_sdf(
                            smiles,
                            single,
                            seed=self.cfg.seed,
                            max_attempts=self.cfg.embed_max_attempts,
                            ff_max_iters=self.cfg.ff_max_iters,
                        )
                    except LigandPrepError as exc:
                        results[smiles] = OracleResult(
                            smiles=smiles,
                            score=None,
                            backend=self.name,
                            backend_version=version,
                            status="prep_failed",
                            error=str(exc),
                        )
                        continue

                    mol = next(iter(Chem.SDMolSupplier(str(single), removeHs=False)), None)
                    if mol is None:
                        results[smiles] = OracleResult(
                            smiles=smiles,
                            score=None,
                            backend=self.name,
                            backend_version=version,
                            status="prep_failed",
                            error="conformer SDF could not be re-read",
                        )
                        continue
                    mol.SetProp("_Name", tag)
                    tag_to_smiles[tag] = smiles
                    writer.write(mol)
            finally:
                writer.close()

            if not tag_to_smiles:
                return [results[s] for s in smiles_list]

            command = [
                self._binary(),
                "--receptor", str(self._receptor()),
                "--ligand", str(batch_sdf),
                "--center_x", str(self.target.center[0]),
                "--center_y", str(self.target.center[1]),
                "--center_z", str(self.target.center[2]),
                "--size_x", str(self.target.size[0]),
                "--size_y", str(self.target.size[1]),
                "--size_z", str(self.target.size[2]),
                "--exhaustiveness", str(self.cfg.exhaustiveness),
                "--num_modes", str(self.cfg.num_modes),
                "--seed", str(self.cfg.seed),
                "--cpu", "1",
                "--out", str(out_sdf),
            ]

            # The timeout scales with the batch, or a large batch would always
            # trip a per-molecule timeout.
            timeout_s = self.cfg.timeout_s * max(1, len(tag_to_smiles))
            failure: tuple[str, str] | None = None
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=timeout_s, check=False
                )
                if completed.returncode != 0:
                    failure = (
                        "dock_failed",
                        f"smina exit {completed.returncode}: {completed.stderr.strip()[:400]}",
                    )
            except subprocess.TimeoutExpired:
                failure = ("timeout", f"smina batch exceeded {timeout_s}s")
            except OSError as exc:
                failure = ("dock_failed", str(exc))

            elapsed = round((perf_counter() - start) / max(1, len(tag_to_smiles)), 3)

            if failure is not None:
                status, message = failure
                for tag, smiles in tag_to_smiles.items():
                    results[smiles] = OracleResult(
                        smiles=smiles,
                        score=None,
                        backend=self.name,
                        backend_version=version,
                        status=status,  # type: ignore[arg-type]
                        error=message,
                        wall_clock_s=elapsed,
                    )
                return [results[s] for s in smiles_list]

            modes = parse_batch_output_all_modes(out_sdf)
            for tag, smiles in tag_to_smiles.items():
                if modes.get(tag):
                    affinities = modes[tag]
                    results[smiles] = OracleResult(
                        smiles=smiles,
                        score=min(affinities),
                        backend=self.name,
                        backend_version=version,
                        status="ok",
                        wall_clock_s=elapsed,
                        extra={
                            "all_mode_affinities": affinities,
                            "batch_size": len(tag_to_smiles),
                        },
                    )
                else:
                    # smina omits ligands it could not dock. Matching by name
                    # means this is detected rather than shifting scores onto
                    # the wrong molecules.
                    results[smiles] = OracleResult(
                        smiles=smiles,
                        score=None,
                        backend=self.name,
                        backend_version=version,
                        status="dock_failed",
                        error="ligand absent from smina batch output",
                        wall_clock_s=elapsed,
                    )

        return [results[s] for s in smiles_list]



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

# --------------------------------------------------------------------------
# Batched docking
# --------------------------------------------------------------------------
#
# Measured on CI: docking one ligand per smina process costs about 11s per
# molecule, and shrinking the search box by 2.4x changed nothing. The cost is
# therefore not search, it is the fixed per-invocation work of reading and
# atom-typing the receptor and building grid maps, paid once per molecule.
#
# Batching writes many ligands into one multi-model SDF and runs smina once
# over it, so that setup is paid once per batch instead.
#
# Results are matched back to molecules by the ligand NAME written into the
# SDF, never by output order. smina omits ligands it fails to dock, so
# position-based matching would silently shift every subsequent score onto
# the wrong molecule, which is the worst possible failure here: it would look
# like a working pipeline producing wrong numbers.

_LIGAND_TAG = "DSAL_LIGAND_{index:06d}"
_AFFINITY_PROP = "minimizedAffinity"


def parse_batch_output_all_modes(path: Path) -> dict[str, list[float]]:
    """Map ligand name to every pose affinity smina wrote for it.

    Reads ``minimizedAffinity`` from the output SDF rather than the stdout
    affinity table, because the table is rounded to one decimal place while
    the SDF property carries full precision. Kept separate from the
    subprocess plumbing so it can be unit-tested against a hand-built SDF
    with no smina binary present.
    """
    scores: dict[str, list[float]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return scores

    # sanitize=False: these are docked poses, and a pose that trips RDKit's
    # sanitiser still carries a perfectly good affinity we should not discard.
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        if not mol.HasProp("_Name") or not mol.HasProp(_AFFINITY_PROP):
            continue
        name = mol.GetProp("_Name")
        try:
            affinity = float(mol.GetProp(_AFFINITY_PROP))
        except (ValueError, TypeError):
            continue
        scores.setdefault(name, []).append(affinity)
    return scores


def parse_batch_output_sdf(path: Path) -> dict[str, float]:
    """Map ligand name to its best (most negative) affinity."""
    return {name: min(vals) for name, vals in parse_batch_output_all_modes(path).items() if vals}
