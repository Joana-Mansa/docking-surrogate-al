"""Tests for batched docking.

Batching exists because smina's per-molecule cost is dominated by reading and
atom-typing the receptor on every invocation, not by search: measured on CI,
one ligand per process cost about 11s each, and cutting the search box by
2.4x changed nothing.

The dangerous failure mode of batching is misalignment. smina omits ligands
it could not dock, so matching results to molecules by output *position*
would shift every subsequent score onto the wrong molecule and produce a
pipeline that looks healthy while reporting wrong numbers. These tests pin
down that results are matched by ligand name, that a batch always returns one
result per input, and that one bad ligand cannot discard its batch.

The tests that need a real smina binary are marked ``requires_smina`` and
skip when it is absent, so the suite stays runnable on a machine without it.
"""

from __future__ import annotations

import shutil

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from docking_surrogate_al.config import OracleConfig, TargetConfig
from docking_surrogate_al.oracle.base import DockingOracle, OracleResult
from docking_surrogate_al.oracle.runner import chunk
from docking_surrogate_al.oracle.smina_backend import (
    SminaOracle,
    parse_batch_output_sdf,
)

SMINA_AVAILABLE = shutil.which("smina") is not None
requires_smina = pytest.mark.skipif(not SMINA_AVAILABLE, reason="smina not on PATH")


# --- chunking -------------------------------------------------------------


def test_chunk_splits_without_losing_or_duplicating():
    items = [f"m{i}" for i in range(23)]
    batches = chunk(items, 5)
    assert [len(b) for b in batches] == [5, 5, 5, 5, 3]
    flat = [s for b in batches for s in b]
    assert flat == items


def test_chunk_handles_edges():
    assert chunk([], 5) == []
    assert chunk(["a"], 5) == [("a",)]
    # size 0 must not produce an infinite loop or empty batches
    assert chunk(["a", "b"], 0) == [("a",), ("b",)]


# --- output parsing -------------------------------------------------------


def _write_sdf(path, entries):
    """Write a multi-model SDF with (name, affinity) entries."""
    writer = Chem.SDWriter(str(path))
    try:
        for name, affinity in entries:
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(mol, randomSeed=0)
            mol.SetProp("_Name", name)
            if affinity is not None:
                mol.SetProp("minimizedAffinity", str(affinity))
            writer.write(mol)
    finally:
        writer.close()


def test_parse_batch_output_maps_names_to_scores(tmp_path):
    path = tmp_path / "out.sdf"
    _write_sdf(path, [("LIG_A", -7.1), ("LIG_B", -6.4), ("LIG_C", -8.9)])
    assert parse_batch_output_sdf(path) == {"LIG_A": -7.1, "LIG_B": -6.4, "LIG_C": -8.9}


def test_parse_batch_output_keeps_the_best_pose_per_ligand(tmp_path):
    """num_modes > 1 writes several poses per ligand; the best one wins."""
    path = tmp_path / "out.sdf"
    _write_sdf(path, [("LIG_A", -7.1), ("LIG_A", -8.5), ("LIG_A", -6.0)])
    assert parse_batch_output_sdf(path) == {"LIG_A": -8.5}


def test_parse_batch_output_skips_entries_without_an_affinity(tmp_path):
    path = tmp_path / "out.sdf"
    _write_sdf(path, [("LIG_A", -7.1), ("LIG_B", None)])
    parsed = parse_batch_output_sdf(path)
    assert parsed == {"LIG_A": -7.1}
    assert "LIG_B" not in parsed


def test_parse_batch_output_on_missing_or_empty_file(tmp_path):
    assert parse_batch_output_sdf(tmp_path / "nope.sdf") == {}
    empty = tmp_path / "empty.sdf"
    empty.write_text("", encoding="utf-8")
    assert parse_batch_output_sdf(empty) == {}


# --- the dock_many contract ----------------------------------------------


class _CountingOracle(DockingOracle):
    """Minimal oracle used to check the default dock_many implementation."""

    name = "counting"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def version(self) -> str:
        return "1.0"

    def settings_fingerprint(self) -> str:
        return "fp"

    def prepare(self) -> dict:
        return {}

    def dock(self, smiles: str) -> OracleResult:
        self.calls.append(smiles)
        return OracleResult(
            smiles=smiles,
            score=-7.0,
            backend=self.name,
            backend_version="1.0",
            status="ok",
        )


def test_default_dock_many_loops_and_preserves_order():
    """A backend that cannot batch still satisfies the contract."""
    oracle = _CountingOracle()
    smiles = ["CCO", "CCN", "CCC"]
    results = oracle.dock_many(smiles)
    assert [r.smiles for r in results] == smiles
    assert oracle.calls == smiles


def test_dock_many_returns_one_result_per_input_with_bad_ligands(monkeypatch):
    """An unembeddable ligand is reported, not dropped, and does not raise."""
    oracle = SminaOracle(OracleConfig(ligand_batch_size=4), TargetConfig())
    monkeypatch.setattr(SminaOracle, "version", lambda self: "smina 1.0")

    # Every ligand fails preparation, so smina is never invoked and we can
    # assert the batch still accounts for all inputs.
    def always_fail(*args, **kwargs):
        from docking_surrogate_al.oracle.prepare import LigandPrepError

        raise LigandPrepError("synthetic failure")

    monkeypatch.setattr(
        "docking_surrogate_al.oracle.smina_backend.generate_conformer_sdf", always_fail
    )

    smiles = ["CCO", "not_a_molecule", "CCN"]
    results = oracle.dock_many(smiles)
    assert [r.smiles for r in results] == smiles
    assert all(r.status == "prep_failed" for r in results)
    assert all(r.score is None for r in results)


def test_dock_many_on_empty_and_single_inputs(monkeypatch):
    oracle = SminaOracle(OracleConfig(), TargetConfig())
    monkeypatch.setattr(SminaOracle, "version", lambda self: "smina 1.0")
    assert oracle.dock_many([]) == []

    monkeypatch.setattr(
        SminaOracle,
        "dock",
        lambda self, s: OracleResult(
            smiles=s, score=-5.0, backend="smina", backend_version="1.0", status="ok"
        ),
    )
    single = oracle.dock_many(["CCO"])
    assert len(single) == 1 and single[0].score == -5.0


# --- the assumption batching rests on ------------------------------------


@requires_smina
@pytest.mark.slow
def test_batching_does_not_change_scores():
    """Batched and unbatched docking must agree on the same molecules.

    This is the assumption that makes ligand_batch_size safe to leave out of
    the cache fingerprint: if batching changed scores, cached results from
    different batch sizes could not be mixed, and the whole optimisation
    would be unsound. It needs a real smina binary, so it runs in CI rather
    than on a developer machine without one.
    """
    molecules = [
        "CCOc1ccccc1C(=O)O",
        "NC(=O)c1ccccc1",
        "Cc1c[nH]c2ccccc12",
        "c1ccc2ncccc2c1",
    ]
    cfg = OracleConfig(exhaustiveness=1, num_modes=1, seed=0)
    oracle = SminaOracle(cfg, TargetConfig(size=(14.0, 14.0, 14.0)))
    oracle.prepare()

    batched = oracle.dock_many(molecules)
    one_at_a_time = [oracle.dock(s) for s in molecules]

    assert [r.smiles for r in batched] == molecules
    for b, s in zip(batched, one_at_a_time, strict=True):
        assert b.status == s.status, f"{b.smiles}: {b.status} vs {s.status}"
        if b.ok and s.ok:
            assert b.score == pytest.approx(s.score, abs=0.01), (
                f"{b.smiles}: batched {b.score} vs unbatched {s.score}. "
                "If this fails, ligand_batch_size must be part of the cache "
                "settings fingerprint."
            )
