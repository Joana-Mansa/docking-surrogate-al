"""Acquisition strategies.

Design note
-----------
Three strategies, and the third is the point of the experiment.

* **uncertainty**: take the unlabelled molecules with the largest predictive
  standard deviation. This is pure exploration: it asks for the labels the
  model is least able to guess, which is what improves the *model*.
* **greedy**: take the molecules with the best (most negative) predicted
  score. Pure exploitation: it asks for the labels most likely to be good,
  which is what improves the *campaign* in the short term.
* **random**: uniform sampling from the unlabelled pool.

Random is mandatory and is not a formality. With a pool of a few thousand
molecules and a top-100 target, random selection recovers a predictable
fraction of the top-100 simply because it labels a fixed share of the pool,
and that fraction is often surprisingly competitive early on. Any claim that
active learning helps has to beat this curve, not just rise.

A note on what uncertainty sampling optimises
---------------------------------------------
Uncertainty sampling is *not* trying to find good molecules; it is trying to
build a good model. On a top-k recall curve it can therefore lose to greedy
while being the better surrogate, because greedy spends every call on the
region the metric scores. Both curves are reported, and the README does not
collapse them into a single "active learning wins" claim.

Ties and determinism
--------------------
Ties are broken by a seeded shuffle before sorting, not by index order. With
an untrained model early in a campaign many predictions are near-identical,
and index-order tie-breaking would systematically favour whichever molecules
happened to be first in the pool file.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

STRATEGIES = ("uncertainty", "greedy", "random")


class UnknownStrategyError(ValueError):
    """Raised for an unrecognised acquisition strategy."""


def select_batch(
    strategy: str,
    *,
    candidate_indices: Sequence[int],
    predicted_mean: np.ndarray | None,
    predicted_sigma: np.ndarray | None,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose the next molecules to send to the oracle.

    Args:
        strategy: One of ``"uncertainty"``, ``"greedy"`` or ``"random"``.
        candidate_indices: Pool indices of unlabelled molecules.
        predicted_mean: Predictions aligned with ``candidate_indices``.
            Required by ``greedy``.
        predicted_sigma: Predictive std aligned with ``candidate_indices``.
            Required by ``uncertainty``.
        batch_size: How many molecules to select.
        rng: Seeded generator, used for sampling and tie-breaking.

    Returns:
        Pool indices of the selected molecules.
    """
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.size == 0:
        return np.array([], dtype=np.int64)

    take = int(min(batch_size, candidates.size))
    if take < batch_size:
        logger.info(
            "Only %d unlabelled molecule(s) remain; selecting all of them "
            "instead of the requested %d.",
            take,
            batch_size,
        )

    if strategy == "random":
        return np.sort(rng.choice(candidates, size=take, replace=False))

    if strategy == "uncertainty":
        scores = _require(predicted_sigma, candidates, "uncertainty", "predicted_sigma")
        # Largest sigma first.
        ordering = _ordered(-scores, rng)
    elif strategy == "greedy":
        scores = _require(predicted_mean, candidates, "greedy", "predicted_mean")
        # Most negative predicted score first.
        ordering = _ordered(scores, rng)
    else:
        raise UnknownStrategyError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    return np.sort(candidates[ordering[:take]])


def _require(
    values: np.ndarray | None, candidates: np.ndarray, strategy: str, field: str
) -> np.ndarray:
    if values is None:
        raise ValueError(f"strategy {strategy!r} requires {field}")
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size != candidates.size:
        raise ValueError(
            f"{field} has {array.size} values but there are {candidates.size} candidates"
        )
    return array


def _ordered(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return indices sorting ``values`` ascending, with random tie-breaking."""
    shuffled = rng.permutation(values.size)
    order = np.argsort(values[shuffled], kind="stable")
    return shuffled[order]


def initial_selection(
    pool_size: int, n_initial: int, rng: np.random.Generator
) -> np.ndarray:
    """Pick the starting labelled set, uniformly at random.

    Every strategy starts from the *same* random set for a given seed, so the
    curves differ only in what they acquired afterwards. Giving each strategy
    its own starting set would confound the comparison with starting luck.
    """
    if n_initial > pool_size:
        raise ValueError(f"al.initial_size={n_initial} exceeds the pool size {pool_size}")
    return np.sort(rng.choice(pool_size, size=n_initial, replace=False))
