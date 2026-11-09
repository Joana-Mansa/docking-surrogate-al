"""Tests for scaffold and random splitting.

The central assertion here is that no Bemis-Murcko scaffold appears in both
train and test. A leaky split is the failure mode that would most flatter the
headline result while being invisible in the metrics, so it is asserted
directly rather than inferred from a performance number looking plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from docking_surrogate_al.data.splits import (
    assert_no_scaffold_overlap,
    group_by_scaffold,
    make_split,
    murcko_scaffold,
    random_split,
    scaffold_split,
)

# Four scaffold families, several analogues each: benzamide, indole,
# quinoline and piperazine-benzene, plus some acyclic compounds.
MOLECULES = [
    # benzamide analogues
    "NC(=O)c1ccccc1",
    "CNC(=O)c1ccccc1",
    "CCNC(=O)c1ccccc1",
    "CCCNC(=O)c1ccccc1",
    "NC(=O)c1ccc(C)cc1",
    "NC(=O)c1ccc(Cl)cc1",
    # indoles
    "c1ccc2[nH]ccc2c1",
    "Cc1c[nH]c2ccccc12",
    "CCc1c[nH]c2ccccc12",
    "OCc1c[nH]c2ccccc12",
    # quinolines
    "c1ccc2ncccc2c1",
    "Cc1ccc2ncccc2c1",
    "Clc1ccc2ncccc2c1",
    # piperazinyl benzenes
    "c1ccc(N2CCNCC2)cc1",
    "Cc1ccc(N2CCNCC2)cc1",
    "Fc1ccc(N2CCNCC2)cc1",
    # acyclic
    "CCCCCCCCO",
    "CCCCCCCCN",
    "CCCCCCCC(=O)O",
    "CCCCCCCCC",
]


def test_murcko_scaffold_groups_analogues():
    """Substituted benzamides share one scaffold."""
    assert murcko_scaffold("NC(=O)c1ccccc1") == murcko_scaffold("CNC(=O)c1ccc(Cl)cc1")


def test_acyclic_molecules_share_the_empty_scaffold():
    """Compounds with no ring system group together, not apart."""
    assert murcko_scaffold("CCCCCCCCO") == ""
    assert murcko_scaffold("CCCCCCCCN") == ""


def test_scaffold_split_has_no_overlap():
    """The core guarantee: no scaffold in more than one partition."""
    split = scaffold_split(MOLECULES, test_frac=0.2, val_frac=0.1)
    assert_no_scaffold_overlap(MOLECULES, split)

    train_scaffolds = {murcko_scaffold(MOLECULES[i]) for i in split.train}
    test_scaffolds = {murcko_scaffold(MOLECULES[i]) for i in split.test}
    assert train_scaffolds.isdisjoint(test_scaffolds)


def test_scaffold_split_partitions_every_molecule_exactly_once():
    split = scaffold_split(MOLECULES, test_frac=0.2, val_frac=0.1)
    combined = np.concatenate([split.train, split.val, split.test])
    assert sorted(combined.tolist()) == list(range(len(MOLECULES)))


def test_scaffold_split_is_deterministic():
    """Repeated calls give identical partitions; no hidden RNG."""
    first = scaffold_split(MOLECULES)
    second = scaffold_split(MOLECULES)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.test, second.test)


def test_scaffold_split_does_not_depend_on_input_order():
    """Shuffling the pool does not change which scaffolds are held out."""
    shuffled_order = list(np.random.default_rng(7).permutation(len(MOLECULES)))
    shuffled = [MOLECULES[i] for i in shuffled_order]

    original = scaffold_split(MOLECULES)
    reordered = scaffold_split(shuffled)

    original_test = {MOLECULES[i] for i in original.test}
    reordered_test = {shuffled[i] for i in reordered.test}
    assert original_test == reordered_test


def test_random_split_is_seeded_and_reproducible():
    first = random_split(MOLECULES, seed=3)
    second = random_split(MOLECULES, seed=3)
    different = random_split(MOLECULES, seed=4)

    assert np.array_equal(first.test, second.test)
    assert not np.array_equal(first.test, different.test)


def test_random_split_leaks_scaffolds_where_scaffold_split_does_not():
    """The optimism of random splitting is real and demonstrable.

    This is the comparison the evaluation section exists to make, so it is
    pinned down as a test rather than left as an assertion in the README.
    """
    leaked = False
    for seed in range(10):
        split = random_split(MOLECULES, test_frac=0.3, val_frac=0.1, seed=seed)
        train_scaffolds = {murcko_scaffold(MOLECULES[i]) for i in split.train}
        test_scaffolds = {murcko_scaffold(MOLECULES[i]) for i in split.test}
        if train_scaffolds & test_scaffolds:
            leaked = True
            break
    assert leaked, "expected at least one random split to share a scaffold"


def test_assert_no_scaffold_overlap_detects_a_leak():
    """The guard actually fires when handed a leaky split."""
    split = random_split(MOLECULES, test_frac=0.3, val_frac=0.1, seed=0)
    groups = group_by_scaffold(MOLECULES)
    biggest = max(groups.values(), key=len)

    # Force a leak: put one member of the largest family on each side.
    leaky = type(split)(
        train=np.array([biggest[0]]),
        val=np.array([], dtype=np.int64),
        test=np.array([biggest[1]]),
        kind="scaffold",
    )
    with pytest.raises(AssertionError, match="scaffold leak"):
        assert_no_scaffold_overlap(MOLECULES, leaky)


def test_make_split_dispatch_and_validation():
    assert make_split("scaffold", MOLECULES, 0.2, 0.1, 0).kind == "scaffold"
    assert make_split("random", MOLECULES, 0.2, 0.1, 0).kind == "random"
    with pytest.raises(ValueError, match="unknown split kind"):
        make_split("kmeans", MOLECULES, 0.2, 0.1, 0)


def test_rejects_fractions_that_leave_no_training_data():
    with pytest.raises(ValueError, match="leave molecules for training"):
        scaffold_split(MOLECULES, test_frac=0.8, val_frac=0.3)
