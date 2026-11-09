"""Molecule standardisation.

Design note
-----------
Standardisation is applied before anything else touches a molecule, because
the canonical SMILES produced here is the *cache key* for the docking oracle.
If standardisation were not deterministic and idempotent, the same compound
could be docked twice under two keys, which would both waste oracle calls and
silently leak information between train and test splits.

The pipeline, in order:

1. Parse, with sanitisation. Unparseable input is dropped, not repaired.
2. Strip salts and solvents, keeping the largest organic fragment. Docking a
   counter-ion is meaningless, and a sodium atom in the graph is noise.
3. Neutralise charges where a neutral form exists, using the standard
   uncharger. Protonation state is a real physical property, but it is one the
   oracle assigns at prep time, so carrying input charges through would mean
   mixing two conventions.
4. Reject molecules outside the configured heavy-atom window, which removes
   both fragments too small to dock meaningfully and outliers whose conformer
   generation would dominate runtime.
5. Emit canonical, isomeric SMILES.
6. Deduplicate on the canonical SMILES.

Every rejection is counted and the tally is written to the run manifest, so a
shrinking dataset is visible rather than mysterious.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# RDKit writes parse failures straight to stderr otherwise.
RDLogger.DisableLog("rdApp.*")

# Elements docking can actually handle, and which the featuriser knows about.
ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B"})


@dataclass(frozen=True)
class StandardisationStats:
    """Tally of what happened to the input set."""

    input_count: int
    unparseable: int
    empty_after_salt_strip: int
    disallowed_element: int
    size_filtered: int
    duplicates: int
    output_count: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable mapping."""
        return {
            "input_count": self.input_count,
            "unparseable": self.unparseable,
            "empty_after_salt_strip": self.empty_after_salt_strip,
            "disallowed_element": self.disallowed_element,
            "size_filtered": self.size_filtered,
            "duplicates": self.duplicates,
            "output_count": self.output_count,
        }


_SALT_REMOVER = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardise_smiles(
    smiles: str,
    *,
    strip_salts: bool = True,
    neutralise: bool = True,
    min_heavy_atoms: int = 8,
    max_heavy_atoms: int = 60,
) -> tuple[str | None, str]:
    """Standardise one SMILES string.

    Args:
        smiles: Input SMILES, in any form.
        strip_salts: Keep only the largest fragment.
        neutralise: Remove formal charges where a neutral form exists.
        min_heavy_atoms: Reject molecules with fewer heavy atoms.
        max_heavy_atoms: Reject molecules with more heavy atoms.

    Returns:
        ``(canonical_smiles, reason)``. On success ``reason`` is ``"ok"``; on
        rejection the SMILES is ``None`` and ``reason`` names the filter that
        rejected it, which is what the stats tally counts.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "unparseable"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparseable"

    if strip_salts:
        mol = _SALT_REMOVER.choose(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None, "empty_after_salt_strip"

    if neutralise:
        try:
            mol = _UNCHARGER.uncharge(mol)
        except Chem.rdchem.MolSanitizeException:
            return None, "unparseable"
        if mol is None:
            return None, "unparseable"

    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.MolSanitizeException, ValueError):
        return None, "unparseable"

    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if not symbols <= ALLOWED_ELEMENTS:
        return None, "disallowed_element"

    heavy = mol.GetNumHeavyAtoms()
    if heavy < min_heavy_atoms or heavy > max_heavy_atoms:
        return None, "size_filtered"

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not canonical:
        return None, "unparseable"
    return canonical, "ok"


def standardise_frame(
    frame: pd.DataFrame,
    *,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    strip_salts: bool = True,
    neutralise: bool = True,
    min_heavy_atoms: int = 8,
    max_heavy_atoms: int = 60,
) -> tuple[pd.DataFrame, StandardisationStats]:
    """Standardise and deduplicate a dataframe of molecules.

    The first occurrence of each canonical SMILES is kept, so the output order
    is a deterministic function of the input order.

    Returns:
        ``(frame, stats)`` where ``frame`` has columns ``canonical_smiles``,
        ``input_smiles`` and, if present, the identifier column, plus a
        ``mol_weight`` and ``n_heavy_atoms`` column for downstream reporting.
    """
    if smiles_column not in frame.columns:
        raise KeyError(f"input has no {smiles_column!r} column; got {list(frame.columns)}")

    reasons: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for record in frame.to_dict("records"):
        canonical, reason = standardise_smiles(
            str(record[smiles_column]),
            strip_salts=strip_salts,
            neutralise=neutralise,
            min_heavy_atoms=min_heavy_atoms,
            max_heavy_atoms=max_heavy_atoms,
        )
        reasons[reason] += 1
        if canonical is None:
            continue
        if canonical in seen:
            reasons["duplicates"] += 1
            continue
        seen.add(canonical)

        mol = Chem.MolFromSmiles(canonical)
        row: dict[str, object] = {
            "canonical_smiles": canonical,
            "input_smiles": record[smiles_column],
            "mol_weight": round(Descriptors.MolWt(mol), 3),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        }
        if id_column and id_column in record:
            row[id_column] = record[id_column]
        rows.append(row)

    stats = StandardisationStats(
        input_count=len(frame),
        unparseable=reasons["unparseable"],
        empty_after_salt_strip=reasons["empty_after_salt_strip"],
        disallowed_element=reasons["disallowed_element"],
        size_filtered=reasons["size_filtered"],
        duplicates=reasons["duplicates"],
        output_count=len(rows),
    )
    logger.info(
        "Standardised %d -> %d molecules (%d unparseable, %d bad element, "
        "%d size-filtered, %d duplicates).",
        stats.input_count,
        stats.output_count,
        stats.unparseable,
        stats.disallowed_element,
        stats.size_filtered,
        stats.duplicates,
    )
    columns = ["canonical_smiles", "input_smiles", "mol_weight", "n_heavy_atoms"]
    if id_column and id_column in frame.columns:
        columns.insert(0, id_column)
    # Pass columns explicitly so an empty result still has the right schema.
    out = pd.DataFrame(rows, columns=columns)
    return out, stats
