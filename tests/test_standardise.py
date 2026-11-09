"""Tests for molecule standardisation."""

from __future__ import annotations

import pandas as pd
import pytest

from docking_surrogate_al.data.standardise import standardise_frame, standardise_smiles


def test_canonicalises_equivalent_smiles():
    """Different writings of the same molecule collapse to one string."""
    kekule, aromatic = "C1=CC=CC=C1C(=O)NCCO", "OCCNC(=O)c1ccccc1"
    first, _ = standardise_smiles(kekule)
    second, _ = standardise_smiles(aromatic)
    assert first is not None
    assert first == second


def test_is_idempotent():
    """Standardising an already-standard SMILES changes nothing.

    This matters because the canonical SMILES is the docking cache key: if it
    were not idempotent, the same molecule could be docked twice.
    """
    once, _ = standardise_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert once is not None
    twice, _ = standardise_smiles(once)
    assert twice == once


def test_strips_salt_to_largest_fragment():
    """A hydrochloride salt reduces to its organic cation/base."""
    result, reason = standardise_smiles("CCCCN(CCCC)CCCC.Cl")
    assert reason == "ok"
    assert result is not None
    assert "Cl" not in result
    assert "." not in result


def test_neutralises_charges():
    """A carboxylate is returned as the neutral acid."""
    result, reason = standardise_smiles("CCCCCCCC(=O)[O-]", neutralise=True)
    assert reason == "ok"
    assert result is not None
    assert "[O-]" not in result


def test_rejects_unparseable_smiles():
    result, reason = standardise_smiles("this is not a molecule")
    assert result is None
    assert reason == "unparseable"


def test_rejects_empty_input():
    assert standardise_smiles("") == (None, "unparseable")


def test_rejects_disallowed_elements():
    """Metals the oracle and featuriser cannot handle are dropped."""
    result, reason = standardise_smiles("[Pt](Cl)(Cl)(N)N")
    assert result is None
    assert reason in {"disallowed_element", "unparseable"}


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [("CCO", "size_filtered"), ("C" * 200, "size_filtered")],
)
def test_size_filters(smiles: str, expected: str):
    """Molecules outside the heavy-atom window are rejected."""
    result, reason = standardise_smiles(smiles, min_heavy_atoms=8, max_heavy_atoms=60)
    assert result is None
    assert reason == expected


def test_frame_deduplicates_and_counts():
    """Duplicates collapse and the stats tally adds up."""
    frame = pd.DataFrame(
        {
            "chembl_id": ["A", "B", "C", "D"],
            "smiles": [
                "CC(=O)Oc1ccccc1C(=O)O",
                "O=C(C)Oc1ccccc1C(=O)O",  # the same molecule written differently
                "not_a_molecule",
                "CCCCCCCCCCO",
            ],
        }
    )
    out, stats = standardise_frame(frame, smiles_column="smiles", id_column="chembl_id")

    assert stats.input_count == 4
    assert stats.unparseable == 1
    assert stats.duplicates == 1
    assert stats.output_count == 2
    assert len(out) == 2
    assert out["canonical_smiles"].is_unique
    assert "chembl_id" in out.columns


def test_frame_keeps_schema_when_everything_is_rejected():
    """An all-rejected input still yields a correctly shaped frame."""
    frame = pd.DataFrame({"chembl_id": ["A"], "smiles": ["???"]})
    out, stats = standardise_frame(frame, smiles_column="smiles", id_column="chembl_id")
    assert out.empty
    assert stats.output_count == 0
    assert "canonical_smiles" in out.columns


def test_missing_column_raises():
    with pytest.raises(KeyError):
        standardise_frame(pd.DataFrame({"x": ["CCO"]}), smiles_column="smiles")
