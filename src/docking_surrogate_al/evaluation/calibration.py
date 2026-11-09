"""Calibration of predictive uncertainty.

Design note
-----------
A predictive standard deviation is only useful if it means something. This
module measures whether it does, on both splits, in two complementary ways.

1. Interval coverage (the reliability curve)
--------------------------------------------
Treat each prediction as a Gaussian with mean ``mu`` and standard deviation
``sigma``. For a nominal level ``p``, the central ``p`` interval is
``mu +- z_p * sigma``. A calibrated model has the true value falling inside
that interval for ``p`` of the molecules. Plotting observed against nominal
coverage gives the regression analogue of a classifier reliability diagram:
the diagonal is perfect, below it is overconfident (intervals too narrow),
above it is underconfident.

Expected calibration error is the mean absolute gap between observed and
nominal coverage over the evaluated levels. Unlike the classification ECE
there is no single standard definition for regression, so the one in use is
stated explicitly in the output: equally spaced nominal levels, unweighted
mean absolute deviation. Comparing this number to one computed under a
different definition would be meaningless, which is why it travels with its
own description.

2. Error-vs-uncertainty binning
-------------------------------
Coverage can look fine while the uncertainty carries no *discriminative*
information, if sigma is roughly constant at about the right size. So
molecules are also binned by predicted sigma and the empirical RMSE per bin is
reported. A useful uncertainty is monotone here: the high-sigma bins should
have visibly higher error. Spearman correlation between sigma and absolute
error summarises that in one number.

The distinction matters for this project specifically. Active learning needs
uncertainty that *ranks* (point 2); honest error bars need uncertainty that is
*scaled* correctly (point 1). A model can pass one and fail the other, so both
are reported rather than collapsing them.

Gaussian assumption
-------------------
Both MC-dropout and a small deep ensemble give a mean and a spread, not a full
predictive distribution, and the Gaussian reading of that spread is an
assumption, not a derivation. It is the standard one in this literature and it
is what makes coverage computable at all, but it is why the reliability curve
is the reported evidence rather than a claim of calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CalibrationResult:
    """Coverage curve, ECE and error-vs-uncertainty diagnostics."""

    nominal_levels: np.ndarray
    observed_coverage: np.ndarray
    expected_calibration_error: float
    miscalibration_area: float
    sigma_error_spearman: float
    mean_sigma: float
    rmse: float
    bin_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_mean_sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_rmse: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_counts: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping, including the ECE definition."""
        return {
            "expected_calibration_error": self.expected_calibration_error,
            "ece_definition": (
                "mean |observed - nominal| central-interval coverage over equally "
                "spaced nominal levels, assuming a Gaussian predictive distribution"
            ),
            "miscalibration_area": self.miscalibration_area,
            "sigma_error_spearman": self.sigma_error_spearman,
            "mean_sigma": self.mean_sigma,
            "rmse": self.rmse,
            "nominal_levels": self.nominal_levels.tolist(),
            "observed_coverage": self.observed_coverage.tolist(),
            "bin_mean_sigma": self.bin_mean_sigma.tolist(),
            "bin_rmse": self.bin_rmse.tolist(),
            "bin_counts": self.bin_counts.tolist(),
        }


def coverage_curve(
    y_true: Sequence[float],
    mean: Sequence[float],
    sigma: Sequence[float],
    n_levels: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed vs nominal central-interval coverage.

    Returns:
        ``(nominal_levels, observed_coverage)``, both length ``n_levels``.
    """
    true = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sd = np.asarray(sigma, dtype=np.float64).ravel()
    if not (true.shape == mu.shape == sd.shape):
        raise ValueError("y_true, mean and sigma must have the same shape")
    if true.size == 0:
        raise ValueError("cannot assess calibration on empty arrays")

    # A zero or negative sigma makes coverage undefined; clamp to a tiny
    # positive value so one degenerate prediction cannot void the whole curve.
    sd = np.maximum(sd, 1e-12)

    # Levels from 1/n to 1 - 1/n inclusive of neither 0 nor 1: coverage at 0
    # and 1 is trivially 0 and 1 and would flatter the ECE.
    nominal = np.linspace(1.0 / (n_levels + 1), n_levels / (n_levels + 1), n_levels)
    observed = np.empty_like(nominal)
    z_scores = np.abs(true - mu) / sd
    for i, level in enumerate(nominal):
        # Two-sided central interval: |z| <= z_{(1+level)/2}
        critical = stats.norm.ppf(0.5 * (1.0 + level))
        observed[i] = float(np.mean(z_scores <= critical))
    return nominal, observed


def expected_calibration_error(nominal: np.ndarray, observed: np.ndarray) -> float:
    """Mean absolute gap between observed and nominal coverage."""
    return float(np.mean(np.abs(observed - nominal)))


def miscalibration_area(nominal: np.ndarray, observed: np.ndarray) -> float:
    """Area between the coverage curve and the diagonal, by trapezoid rule.

    Less sensitive to the number of levels than ECE, so it is reported
    alongside as a cross-check that the ECE is not an artefact of binning.
    """
    # np.trapezoid replaced np.trapz in NumPy 2.0; the pinned version has the
    # former, but the fallback keeps this importable under NumPy 1.x too.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(np.abs(observed - nominal), nominal))


def error_vs_uncertainty(
    y_true: Sequence[float],
    mean: Sequence[float],
    sigma: Sequence[float],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin molecules by predicted sigma and report empirical RMSE per bin.

    Bins are *quantile* bins, not equal-width, so every bin holds a similar
    number of molecules. Equal-width bins on a skewed sigma distribution leave
    the top bins nearly empty, and an RMSE computed over two molecules is not
    a measurement.

    Returns:
        ``(bin_edges, bin_mean_sigma, bin_rmse, bin_counts)``.
    """
    true = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sd = np.asarray(sigma, dtype=np.float64).ravel()
    errors = np.abs(true - mu)

    n_bins = max(1, min(n_bins, sd.size))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(sd, quantiles))
    if edges.size < 2:
        # Constant sigma: one bin is the honest answer.
        edges = np.array([sd.min(), sd.min() + 1e-12])

    assignments = np.clip(np.digitize(sd, edges[1:-1], right=False), 0, edges.size - 2)
    n_actual = edges.size - 1
    mean_sigma = np.full(n_actual, np.nan)
    rmse_per_bin = np.full(n_actual, np.nan)
    counts = np.zeros(n_actual, dtype=np.int64)

    for b in range(n_actual):
        mask = assignments == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            mean_sigma[b] = float(sd[mask].mean())
            rmse_per_bin[b] = float(np.sqrt(np.mean(errors[mask] ** 2)))

    return edges, mean_sigma, rmse_per_bin, counts


def assess_calibration(
    y_true: Sequence[float],
    mean: Sequence[float],
    sigma: Sequence[float],
    n_bins: int = 10,
) -> CalibrationResult:
    """Run the full calibration assessment."""
    true = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sd = np.asarray(sigma, dtype=np.float64).ravel()

    nominal, observed = coverage_curve(true, mu, sd, n_levels=n_bins)
    edges, bin_sigma, bin_rmse, counts = error_vs_uncertainty(true, mu, sd, n_bins=n_bins)

    errors = np.abs(true - mu)
    if sd.size > 1 and not np.all(sd == sd[0]) and not np.all(errors == errors[0]):
        rank_corr = float(stats.spearmanr(sd, errors).statistic)
    else:
        rank_corr = float("nan")

    return CalibrationResult(
        nominal_levels=nominal,
        observed_coverage=observed,
        expected_calibration_error=expected_calibration_error(nominal, observed),
        miscalibration_area=miscalibration_area(nominal, observed),
        sigma_error_spearman=rank_corr,
        mean_sigma=float(np.mean(sd)),
        rmse=float(np.sqrt(np.mean((true - mu) ** 2))),
        bin_edges=edges,
        bin_mean_sigma=bin_sigma,
        bin_rmse=bin_rmse,
        bin_counts=counts,
    )
