"""Ligand and receptor preparation for docking.

Design note
-----------
Preparation is separated from the backend because it is where most docking
failures actually happen, and because the choices made here are assumptions
about the physics that deserve to be stated rather than buried in a backend.

Ligands
-------
smina docks a 3D conformer, so each SMILES is embedded once with ETKDGv3 and
relaxed with MMFF94 (falling back to UFF, which covers the elements MMFF does
not parameterise). A *single* starting conformer is used, not an ensemble:
smina's search already samples ligand torsions internally, so supplying many
starting conformers mostly buys redundant search, and the point here is a
consistent, cheap, reproducible input rather than the best achievable pose.
The embedding is seeded, so the same SMILES yields the same conformer.

Hydrogens are added before embedding (geometry needs them) and kept in the
output, because smina's scoring terms are defined over the polar hydrogens.

Receptor
--------
The raw PDB is cleaned by removing waters, the co-crystallised ligand and
other heteroatoms, keeping protein ATOM records only. This is the minimum
defensible preparation and it is deliberately conservative: it does not model
missing loops, assign protonation states at a given pH, or sample side-chain
rotamers. Those are real limitations of the target preparation and are listed
as such in the README, rather than hidden behind an automated "prep" step that
makes arbitrary choices silently.

If Open Babel is available the cleaned PDB is converted to PDBQT, which is the
format smina prefers and which caches the partial charges and atom types so
they are not recomputed for every one of thousands of docking runs. If it is
not available, smina reads the cleaned PDB directly (it links Open Babel
internally); this is slower but gives the same answer, and the fallback is
logged so the difference in runtime is never a mystery.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# Heteroatom residues that are solvent/buffer rather than protein.
_SOLVENT_RESIDUES = frozenset({"HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "MPD", "PEG", "DMS"})


class LigandPrepError(RuntimeError):
    """Raised when a ligand cannot be embedded in 3D."""


def generate_conformer_sdf(
    smiles: str,
    out_path: Path,
    *,
    seed: int = 0,
    max_attempts: int = 10,
    ff_max_iters: int = 500,
) -> None:
    """Embed ``smiles`` in 3D and write it as an SDF.

    Args:
        smiles: Canonical SMILES.
        out_path: Destination ``.sdf`` path.
        seed: Embedding random seed, for reproducible conformers.
        max_attempts: ETKDG retries before giving up.
        ff_max_iters: Force-field minimisation iterations.

    Raises:
        LigandPrepError: If parsing, embedding or writing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise LigandPrepError(f"cannot parse SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = max_attempts * 100
    params.useRandomCoords = False

    if AllChem.EmbedMolecule(mol, params) != 0:
        # Random starting coordinates rescue most of the stubborn cases,
        # typically macrocycles and heavily bridged systems.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise LigandPrepError(f"ETKDG embedding failed after retry: {smiles!r}")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=ff_max_iters)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=ff_max_iters)
    except (RuntimeError, ValueError) as exc:
        # An unminimised but valid embedding still docks; note and continue.
        logger.debug("Force-field minimisation failed for %s: %s", smiles, exc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    try:
        writer.write(mol)
    finally:
        writer.close()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise LigandPrepError(f"SDF write produced no output for {smiles!r}")


def clean_receptor_pdb(raw_pdb: Path, cleaned_pdb: Path) -> dict[str, int]:
    """Strip waters, ligands and other heteroatoms from a PDB file.

    Keeps ATOM records (and their TER/END structure), plus any HETATM that is
    not a known solvent and not the co-crystal ligand. In practice for a
    single-chain structure this leaves the protein alone.

    Returns:
        Counts of kept and dropped records, for logging and the manifest.
    """
    if not raw_pdb.exists():
        raise FileNotFoundError(f"receptor PDB not found: {raw_pdb}")

    kept: list[str] = []
    n_atom = 0
    n_dropped_het = 0
    for line in raw_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        record = line[:6].strip()
        if record == "ATOM":
            # Skip alternate conformations beyond the first.
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            kept.append(line)
            n_atom += 1
        elif record == "HETATM":
            n_dropped_het += 1
        elif record in ("TER", "END"):
            kept.append(line)

    cleaned_pdb.parent.mkdir(parents=True, exist_ok=True)
    cleaned_pdb.write_text("\n".join(kept) + "\n", encoding="utf-8")
    stats = {"atom_records_kept": n_atom, "hetatm_records_dropped": n_dropped_het}
    logger.info("Cleaned receptor %s -> %s: %s", raw_pdb.name, cleaned_pdb.name, stats)
    if n_atom == 0:
        raise ValueError(f"{raw_pdb} contained no ATOM records; is it a protein structure?")
    return stats


def convert_to_pdbqt(cleaned_pdb: Path, pdbqt: Path) -> bool:
    """Convert a cleaned PDB to PDBQT with Open Babel, if available.

    Returns:
        True if the PDBQT was produced, False if Open Babel is unavailable
        (in which case callers should dock against the cleaned PDB instead).
    """
    obabel = shutil.which("obabel")
    if obabel is None:
        logger.warning(
            "Open Babel ('obabel') not found on PATH; smina will read the cleaned "
            "PDB directly. This is slower (atom typing is redone per docking run) "
            "but gives the same scores."
        )
        return False

    pdbqt.parent.mkdir(parents=True, exist_ok=True)
    command = [
        obabel,
        str(cleaned_pdb),
        "-O",
        str(pdbqt),
        "-xr",  # rigid receptor: no torsion tree
        "-p",
        "7.4",  # add hydrogens for pH 7.4
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Open Babel conversion failed (%s); falling back to PDB.", exc)
        return False

    if result.returncode != 0 or not pdbqt.exists() or pdbqt.stat().st_size == 0:
        logger.warning(
            "Open Babel returned %d and produced no usable PDBQT; falling back to PDB. "
            "stderr: %s",
            result.returncode,
            result.stderr.strip()[:400],
        )
        return False

    logger.info("Prepared receptor PDBQT at %s (Open Babel, pH 7.4).", pdbqt)
    return True
