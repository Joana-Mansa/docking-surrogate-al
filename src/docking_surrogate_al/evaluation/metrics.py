"""Regression and ranking metrics.

Sign convention
---------------
Docking scores are in kcal/mol and **more negative is better**, following
smina. "Top" therefore always means most negative, and every ranking in this
module sorts ascending. Getting this backwards silently inverts enrichment,
so it is asserted in the test suite with a hand-checked example.

Why these metrics
-----------------
RMSE answers "how close is the surrogate to the oracle in absolute terms",
which is the fidelity question. But a surrogate used for screening is never
asked for an absolute number; it is asked to *rank*, and then only the top of
the ranking is ever looked at. So:

* **Spearman rho** measures whole-pool rank agreement.
* **Enrichment factor** at 1% and 5% measures the top of the ranking, which is
  the part a screening campaign actually consumes.
* **Precision@k** for k in {10, 50, 100} measures the very top, at the scale a
  human follows up by hand.

A model can have good RMSE and useless enrichment (it is accurate in the dense
middle of the distribution and blind at the tail), or poor RMSE and strong
enrichment (it is systematically offset but ranks well). Reporting only one of
them hides which case you are in, so all are reported together.

Enrichment factor, precisely
----------------------------
"Actives" are defined as the top ``fraction`` of molecules by *true* docking
score. Then::

    EF@f = (fraction of the top-f predicted that are true actives) / f

This is the standard hit-rate-over-random form: random selection gives EF = 1,
and perfect ranking gives EF = 1/f (so EF@1% caps at 100). Defining actives by
a percentile of the same pool, rather than an absolute score threshold, keeps
the metric comparable across targets whose score distributions differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


class MetricInputError(ValueError):
    """Raised when metric inputs are unusable."""


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.shape != pred.shape:
        raise MetricInputError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        raise MetricInputError("cannot compute metrics on empty arrays")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise MetricInputError("y_true and y_pred must be finite")
    return true, pred


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error in kcal/mol."""
    true, pred = _validate(np.asarray(y_true), np.asarray(y_pred))
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error in kcal/mol."""
    true, pred = _validate(np.asarray(y_true), np.asarray(y_pred))
    return float(np.mean(np.abs(true - pred)))


def spearman_rho(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation.

    Returns NaN when either input is constant, which is undefined rather than
    zero; a bootstrap resample of a near-constant slice can produce this, and
    reporting it as 0.0 would quietly bias the confidence interval.
    """
    true, pred = _validate(np.asarray(y_true), np.asarray(y_pred))
    if true.size < 2 or np.all(true == true[0]) or np.all(pred == pred[0]):
        return float("nan")
    return float(stats.spearmanr(true, pred).statistic)


def pearson_r(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Pearson correlation coefficient."""
    true, pred = _validate(np.asarray(y_true), np.asarray(y_pred))
    if true.size < 2 or np.all(true == true[0]) or np.all(pred == pred[0]):
        return float("nan")
    return float(stats.pearsonr(true, pred).statistic)


def _top_n(fraction: float, total: int) -> int:
    """Number of molecules in the top ``fraction``, at least one."""
    if not 0.0 < fraction <= 1.0:
        raise MetricInputError(f"fraction must be in (0, 1], got {fraction}")
    return max(1, int(round(total * fraction)))


def enrichment_factor(
    y_true: Sequence[float], y_pred: Sequence[float], fraction: float = 0.01
) -> float:
    """Enrichment factor at the given top fraction.

    Args:
        y_true: True docking scores (more negative is better).
        y_pred: Predicted docking scores.
        fraction: Top fraction to select, e.g. 0.01 for 1%.

    Returns:
        Hit rate in the selected set divided by ``fraction``. 1.0 is random,
        ``1 / fraction`` is perfect.
    """
    true, pred = _validate(np.asarray(y_true), np.asarray(y_pred))
    n = true.size
    k = _top_n(fraction, n)

    # Ascending: most negative first.
    true_actives = set(np.argsort(true, kind="stable")[:k].tolist())
    selected = np.argsort(pred, kind="stable")[:k]
    hits = sum(1 for i in selected.tolist() if i in true_actives)
    return float((hits / k) / fraction)


def precision_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int = 100) -> float:
    """Fraction of the top-k predicted that are in the top-k true.

    ``k`` larger than the pool is clamped to the pool size, which makes the
    metric degrade gracefully on small smoke-test sets instead of raising.
    """
    true, pred = _validate(np.asarray(y_true), np.asarray(y_pred))
    if k < 1:
        raise MetricInputError(f"k must be >= 1, got {k}")
    k = min(k, true.size)
    true_top = set(np.argsort(true, kind="stable")[:k].tolist())
    pred_top = np.argsort(pred, kind="stable")[:k]
    hits = sum(1 for i in pred_top.tolist() if i in true_top)
    return float(hits / k)


def top_k_recall(
    y_true: Sequence[float], selected_indices: Sequence[int], k: int = 100
) -> float:
    """Fraction of the pool's true top-k that ``selected_indices`` contains.

    This is the active-learning headline metric: of the k best molecules in
    the pool, how many has the campaign actually paid the oracle for? It takes
    *selected indices* rather than predictions, because what matters in active
    learning is which molecules were sent to the oracle, not how they ranked.
    """
    true = np.asarray(y_true, dtype=np.float64).ravel()
    if true.size == 0:
        raise MetricInputError("cannot compute recall on an empty pool")
    k = min(k, true.size)
    true_top = set(np.argsort(true, kind="stable")[:k].tolist())
    found = len(true_top & set(int(i) for i in selected_indices))
    return float(found / k)


@dataclass(frozen=True)
class MetricSuite:
    """Configured set of metrics, evaluated together."""

    ef_fractions: tuple[float, ...] = (0.01, 0.05)
    precision_k: tuple[int, ...] = (10, 50, 100)

    def names(self) -> list[str]:
        """Return the metric names this suite produces, in order."""
        names = ["rmse", "mae", "spearman_rho", "pearson_r"]
        names += [f"ef_{f:g}".replace("0.", "") for f in self.ef_fractions]
        names += [f"precision_at_{k}" for k in self.precision_k]
        return names

    def compute(self, y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
        """Compute every metric in the suite."""
        results: dict[str, float] = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "spearman_rho": spearman_rho(y_true, y_pred),
            "pearson_r": pearson_r(y_true, y_pred),
        }
        for fraction in self.ef_fractions:
            key = f"ef_{fraction:g}".replace("0.", "")
            results[key] = enrichment_factor(y_true, y_pred, fraction)
        for k in self.precision_k:
            results[f"precision_at_{k}"] = precision_at_k(y_true, y_pred, k)
        return results

    def describe(self) -> Mapping[str, Any]:
        """Return a manifest-ready description of the suite."""
        return {
            "ef_fractions": list(self.ef_fractions),
            "precision_k": list(self.precision_k),
            "sign_convention": "more negative docking score is better; top = most negative",
            "ef_definition": "actives = top `fraction` by true score; EF = hit_rate / fraction",
        }
