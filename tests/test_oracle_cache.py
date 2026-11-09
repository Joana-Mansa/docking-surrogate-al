"""Tests for the docking cache, oracle plumbing and smina output parsing.

The cache's job is to make expensive work survive interruption, so the tests
simulate interruption directly: a truncated final line, a reopened file, a
changed settings fingerprint.
"""

from __future__ import annotations

import pytest

from docking_surrogate_al.config import OracleConfig, TargetConfig
from docking_surrogate_al.oracle import make_oracle
from docking_surrogate_al.oracle.base import OracleResult
from docking_surrogate_al.oracle.cache import DockingCache
from docking_surrogate_al.oracle.runner import resolve_workers
from docking_surrogate_al.oracle.smina_backend import parse_affinity_table

SMINA_OUTPUT = """\
   _______  _______ _________ _        _______
  (  ____ \\(       )\\__   __/( (    /|(  ___  )
smina is based off AutoDock Vina. Please cite appropriately.

Using random seed: 42

mode |   affinity | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -7.12       0.000      0.000
   2       -6.84       1.923      2.455
   3       -6.01       2.110      3.004

Refine time 1.234
Loop time 5.678
"""


def _result(smiles: str, score: float | None, status: str = "ok") -> OracleResult:
    return OracleResult(
        smiles=smiles,
        score=score,
        backend="smina",
        backend_version="smina 1.0",
        status=status,  # type: ignore[arg-type]
        wall_clock_s=1.0,
    )


def test_parse_affinity_table_reads_every_mode():
    affinities = parse_affinity_table(SMINA_OUTPUT)
    assert affinities == [-7.12, -6.84, -6.01]
    assert min(affinities) == -7.12


def test_parse_affinity_table_on_garbage_returns_nothing():
    assert parse_affinity_table("smina: error while loading shared libraries") == []
    assert parse_affinity_table("") == []


def test_cache_appends_and_reads_back(tmp_path):
    path = tmp_path / "cache.jsonl"
    with DockingCache(path, "fp1") as cache:
        cache.append(_result("CCO", -5.5))
        cache.append(_result("CCN", -6.5))

    assert path.exists()
    reopened = DockingCache(path, "fp1")
    assert len(reopened) == 2
    assert "CCO" in reopened
    assert reopened.get("CCO").score == pytest.approx(-5.5)
    assert reopened.successful() == {"CCO": -5.5, "CCN": -6.5}


def test_cache_survives_a_truncated_final_line(tmp_path):
    """A half-written record from a hard kill is skipped, not fatal."""
    path = tmp_path / "cache.jsonl"
    with DockingCache(path, "fp1") as cache:
        cache.append(_result("CCO", -5.5))
        cache.append(_result("CCN", -6.5))

    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"smiles": "CCC", "score": -7.0, "backe')  # killed mid-write

    reopened = DockingCache(path, "fp1")
    assert len(reopened) == 2
    assert "CCC" not in reopened


def test_cache_partitions_by_settings_fingerprint(tmp_path):
    """Changing the box or exhaustiveness must not reuse old scores."""
    path = tmp_path / "cache.jsonl"
    with DockingCache(path, "box_20A") as cache:
        cache.append(_result("CCO", -5.5))

    other = DockingCache(path, "box_30A")
    assert len(other) == 0
    assert other.missing(["CCO"]) == ["CCO"]

    same = DockingCache(path, "box_20A")
    assert same.missing(["CCO"]) == []


def test_cache_last_record_wins(tmp_path):
    """The file is an append-only log; a re-dock supersedes the old value."""
    path = tmp_path / "cache.jsonl"
    with DockingCache(path, "fp1") as cache:
        cache.append(_result("CCO", -5.5))
        cache.append(_result("CCO", -6.0))
    reopened = DockingCache(path, "fp1")
    assert len(reopened) == 1
    assert reopened.get("CCO").score == pytest.approx(-6.0)


def test_missing_skips_failures_unless_retrying(tmp_path):
    """A molecule that failed prep will fail again; do not retry by default."""
    path = tmp_path / "cache.jsonl"
    with DockingCache(path, "fp1") as cache:
        cache.append(_result("CCO", -5.5))
        cache.append(_result("BAD", None, status="prep_failed"))

    cache = DockingCache(path, "fp1")
    assert cache.missing(["CCO", "BAD", "NEW"]) == ["NEW"]
    assert cache.missing(["CCO", "BAD", "NEW"], retry_failures=True) == ["BAD", "NEW"]
    assert cache.successful() == {"CCO": -5.5}


def test_cache_summary_records_backend_provenance(tmp_path):
    """Every score must be attributable to the oracle that produced it."""
    path = tmp_path / "cache.jsonl"
    with DockingCache(path, "fp1") as cache:
        cache.append(_result("CCO", -5.5))
        cache.append(_result("BAD", None, status="timeout"))

    summary = DockingCache(path, "fp1").summary()
    assert summary["total_records"] == 2
    assert summary["status_counts"] == {"ok": 1, "timeout": 1}
    assert summary["backend_counts"] == {"smina:smina 1.0": 2}
    assert summary["settings_fingerprint"] == "fp1"


def test_resolve_workers_leaves_a_core_free():
    assert resolve_workers(4) == 4
    assert resolve_workers(None) >= 1


def test_make_oracle_dispatch():
    oracle = make_oracle(OracleConfig(backend="smina"), TargetConfig())
    assert oracle.name == "smina"

    with pytest.raises(ValueError, match="unknown oracle.backend"):
        make_oracle(OracleConfig(backend="vina"), TargetConfig())


def test_gnina_backend_refuses_rather_than_pretending():
    """The stub must fail clearly, not return plausible-looking numbers."""
    with pytest.raises(NotImplementedError, match="documented stub"):
        make_oracle(OracleConfig(backend="gnina"), TargetConfig())


def test_smina_fingerprint_changes_with_the_box(monkeypatch):
    """Settings that change a score must change the cache partition."""
    from docking_surrogate_al.oracle.smina_backend import SminaOracle

    monkeypatch.setattr(SminaOracle, "version", lambda self: "smina 1.0")

    base = SminaOracle(OracleConfig(), TargetConfig())
    moved = SminaOracle(OracleConfig(), TargetConfig(size=(30.0, 30.0, 30.0)))
    deeper = SminaOracle(OracleConfig(exhaustiveness=32), TargetConfig())

    assert base.settings_fingerprint() != moved.settings_fingerprint()
    assert base.settings_fingerprint() != deeper.settings_fingerprint()
    assert base.settings_fingerprint() == SminaOracle(
        OracleConfig(), TargetConfig()
    ).settings_fingerprint()
