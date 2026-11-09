"""Train/validation/test splitting.

Design note
-----------
Two splits are produced and always reported side by side, because the gap
between them *is* a result. A random split lets near-identical analogues sit
on both sides of the partition, so the model is partly asked to recall
molecules it has effectively already seen. Reporting only a random split
systematically overstates how well a surrogate will work on new chemistry.

Bemis-Murcko scaffold split
---------------------------
Molecules are grouped by their Murcko scaffold (ring systems plus the linkers
between them, with side chains removed). Whole groups are assigned to exactly
one partition, so no scaffold appears in both train and test. Groups are
assigned largest-first to train, which is the standard deterministic
construction: it puts the common, well-populated chemotypes in train and
leaves the rarer ones for evaluation, which is the harder and more honest
direction.

``scaffold_split`` is deterministic given the molecule order and takes no
seed, by design. A "seeded scaffold split" would either shuffle groups (which
makes the split depend on luck rather than on scaffold frequency) or shuffle
within groups (which does nothing). Where seed-to-seed variation is wanted,
the active-learning loop varies which molecules are *labelled*, not which
scaffolds are held out, so the test set stays fixed across seeds and the
recall curves remain comparable.

Even a scaffold split is optimistic relative to true prospective use: the
scaffolds still come from the same library, prepared the same way, and docked
against the same target with the same box. See the README's Limitations.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitIndices:
    """Index arrays into the molecule pool."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    kind: str

    def sizes(self) -> dict[str, int]:
        """Return partition sizes."""
        return {
            "train": int(self.train.size),
            "val": int(self.val.size),
            "test": int(self.test.size),
        }


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Return the Bemis-Murcko scaffold SMILES for ``smiles``.

    Molecules with no ring system (acyclic compounds) reduce to an empty
    scaffold. They are grouped together under ``""``, which is correct: they
    share the property of having no scaffold, and splitting them apart would
    put structurally trivial analogues on both sides.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES for scaffold: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def group_by_scaffold(smiles_list: list[str]) -> dict[str, list[int]]:
    """Map each scaffold to the indices of the molecules that have it."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, smiles in enumerate(smiles_list):
        groups[murcko_scaffold(smiles)].append(index)
    return dict(groups)


def scaffold_split(
    smiles_list: list[str], test_frac: float = 0.2, val_frac: float = 0.1
) -> SplitIndices:
    """Split by Bemis-Murcko scaffold, largest group first into train.

    Args:
        smiles_list: Canonical SMILES in pool order.
        test_frac: Target fraction of molecules in test.
        val_frac: Target fraction of molecules in validation.

    Returns:
        :class:`SplitIndices` with no scaffold shared between partitions.
    """
    _check_fractions(test_frac, val_frac)
    total = len(smiles_list)
    groups = group_by_scaffold(smiles_list)

    # Sort by size descending, then by scaffold SMILES for a deterministic
    # tie-break independent of dict insertion order.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    n_test_target = int(round(total * test_frac))
    n_val_target = int(round(total * val_frac))

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for _scaffold, indices in ordered:
        if len(test) + len(indices) <= n_test_target:
            test.extend(indices)
        elif len(val) + len(indices) <= n_val_target:
            val.extend(indices)
        else:
            train.extend(indices)

    split = SplitIndices(
        train=np.array(sorted(train), dtype=np.int64),
        val=np.array(sorted(val), dtype=np.int64),
        test=np.array(sorted(test), dtype=np.int64),
        kind="scaffold",
    )
    logger.info(
        "Scaffold split: %s over %d distinct scaffolds.", split.sizes(), len(groups)
    )
    return split


def random_split(
    smiles_list: list[str], test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 0
) -> SplitIndices:
    """Split uniformly at random, as the optimistic comparison point."""
    _check_fractions(test_frac, val_frac)
    total = len(smiles_list)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total)

    n_test = int(round(total * test_frac))
    n_val = int(round(total * val_frac))
    test = permutation[:n_test]
    val = permutation[n_test : n_test + n_val]
    train = permutation[n_test + n_val :]

    split = SplitIndices(
        train=np.sort(train),
        val=np.sort(val),
        test=np.sort(test),
        kind="random",
    )
    logger.info("Random split (seed=%d): %s.", seed, split.sizes())
    return split


def make_split(
    kind: str, smiles_list: list[str], test_frac: float, val_frac: float, seed: int
) -> SplitIndices:
    """Dispatch to the requested split type."""
    if kind == "scaffold":
        return scaffold_split(smiles_list, test_frac=test_frac, val_frac=val_frac)
    if kind == "random":
        return random_split(smiles_list, test_frac=test_frac, val_frac=val_frac, seed=seed)
    raise ValueError(f"unknown split kind {kind!r}; expected 'scaffold' or 'random'")


def assert_no_scaffold_overlap(smiles_list: list[str], split: SplitIndices) -> None:
    """Raise if any scaffold occurs in more than one partition.

    Called after constructing a scaffold split, and asserted directly in the
    test suite, because a leaky split is the failure mode that would most
    flatter the headline result while being invisible in the metrics.
    """
    partitions = {
        "train": {murcko_scaffold(smiles_list[i]) for i in split.train},
        "val": {murcko_scaffold(smiles_list[i]) for i in split.val},
        "test": {murcko_scaffold(smiles_list[i]) for i in split.test},
    }
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        shared = partitions[a] & partitions[b]
        if shared:
            raise AssertionError(
                f"scaffold leak between {a} and {b}: {len(shared)} shared scaffold(s), "
                f"e.g. {sorted(shared)[:3]}"
            )


def _check_fractions(test_frac: float, val_frac: float) -> None:
    if not 0.0 <= test_frac < 1.0 or not 0.0 <= val_frac < 1.0:
        raise ValueError("test_frac and val_frac must each be in [0, 1)")
    if test_frac + val_frac >= 1.0:
        raise ValueError(
            f"test_frac + val_frac must leave molecules for training "
            f"(got {test_frac} + {val_frac})"
        )
