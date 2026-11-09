"""Bootstrap confidence intervals.

Design note
-----------
Every reported metric carries a 95% CI, because a single number on a test set
of a few hundred molecules invites over-reading differences that are inside
the noise. The resampling is *non-parametric and paired*: molecules are
resampled with replacement, and each resample recomputes the metric from the
same (true, predicted) pairs. Resampling molecules rather than residuals is
what makes the interval valid for rank-based metrics like Spearman and
enrichment, which are not functions of the residuals alone.

Percentile intervals are used rather than BCa. BCa corrects for skew and bias
and would be slightly better for enrichment factor, which is bounded and
discrete; the reason it is not used is that it needs a jackknife pass over the
test set per metric, and the honest gain on n in the hundreds did not justify
quadrupling the runtime of every evaluation. The interval is therefore
approximate, and the code says so rather than implying exactness.

Two caveats recorded in the output:

* Enrichment factor at 1% on a small test set is coarse. With 200 test
  molecules the top 1% is two molecules, so EF takes only three values and its
  CI is wide and lumpy. ``n_top`` is reported alongside so a reader can see
  when this is happening.
* NaN resamples (constant slices, which make rank correlation undefined) are
  dropped before taking percentiles, and the count is reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricEstimate:
    """A point estimate with a bootstrap confidence interval."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_resamples: int
    n_valid_resamples: int

    def to_dict(self) -> dict[str, Any]:
        """Return a CSV/JSON-ready mapping."""
        return {
            "metric": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "n_resamples": self.n_resamples,
            "n_valid_resamples": self.n_valid_resamples,
        }

    def format(self, digits: int = 3) -> str:
        """Human-readable ``value [low, high]``."""
        return (
            f"{self.value:.{digits}f} "
            f"[{self.ci_low:.{digits}f}, {self.ci_high:.{digits}f}]"
        )


def bootstrap_metric(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    name: str,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> MetricEstimate:
    """Bootstrap a single metric by resampling molecules with replacement.

    Args:
        metric_fn: Callable taking ``(y_true, y_pred)`` arrays.
        y_true: True docking scores.
        y_pred: Predicted docking scores.
        name: Metric name for the output record.
        n_resamples: Number of bootstrap resamples.
        ci_level: Interval level, e.g. 0.95.
        seed: RNG seed, so intervals are reproducible.

    Returns:
        A :class:`MetricEstimate`.
    """
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")

    point = float(metric_fn(true, pred))
    rng = np.random.default_rng(seed)
    n = true.size

    estimates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            estimates[i] = metric_fn(true[idx], pred[idx])
        except Exception:  # noqa: BLE001 - a degenerate resample is not fatal
            estimates[i] = np.nan

    valid = estimates[np.isfinite(estimates)]
    if valid.size == 0:
        logger.warning("All %d bootstrap resamples of %s were invalid.", n_resamples, name)
        return MetricEstimate(
            name=name,
            value=point,
            ci_low=float("nan"),
            ci_high=float("nan"),
            ci_level=ci_level,
            n_resamples=n_resamples,
            n_valid_resamples=0,
        )

    alpha = (1.0 - ci_level) / 2.0
    low, high = np.percentile(valid, [100 * alpha, 100 * (1 - alpha)])
    return MetricEstimate(
        name=name,
        value=point,
        ci_low=float(low),
        ci_high=float(high),
        ci_level=ci_level,
        n_resamples=n_resamples,
        n_valid_resamples=int(valid.size),
    )


def bootstrap_suite(
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict[str, MetricEstimate]:
    """Bootstrap every metric in ``metric_fns``.

    All metrics share the same resample *indices* (the seed is fixed per
    metric), so their intervals are computed on the same synthetic test sets
    and differences between metrics are not confounded by different draws.
    """
    return {
        name: bootstrap_metric(
            fn,
            y_true,
            y_pred,
            name=name,
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=seed,
        )
        for name, fn in metric_fns.items()
    }


def seed_variance_band(
    values_per_seed: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lower, upper) across seeds for a per-round curve.

    Used for the active-learning figures, where the band is the spread across
    *seeds*, not a bootstrap interval. The two mean different things and are
    never mixed in one figure: seed spread answers "would a rerun look like
    this", a bootstrap CI answers "is this test set big enough to tell".

    The band is mean +- 1 standard deviation across seeds. With five seeds a
    percentile band would be determined entirely by the extremes, so the
    standard deviation is the more stable summary; it is labelled as such in
    the figures.
    """
    array = np.asarray(values_per_seed, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D (seeds x rounds) array, got shape {array.shape}")
    mean = array.mean(axis=0)
    std = array.std(axis=0, ddof=1) if array.shape[0] > 1 else np.zeros_like(mean)
    return mean, mean - std, mean + std
