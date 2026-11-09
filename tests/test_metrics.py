"""Tests for metric implementations.

Metrics are checked against hand-computed values on small arrays, not against
another library, because the thing most likely to be wrong is the *convention*
(more negative is better) rather than the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from docking_surrogate_al.evaluation.bootstrap import bootstrap_metric, seed_variance_band
from docking_surrogate_al.evaluation.metrics import (
    MetricInputError,
    MetricSuite,
    enrichment_factor,
    mae,
    precision_at_k,
    rmse,
    spearman_rho,
    top_k_recall,
)


def test_rmse_and_mae_hand_computed():
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [1.0, -1.0, 2.0, -2.0]
    # squared errors 1, 1, 4, 4 -> mean 2.5 -> sqrt 1.5811...
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2.5))
    assert mae(y_true, y_pred) == pytest.approx(1.5)


def test_rmse_is_zero_for_perfect_predictions():
    values = [-7.1, -8.2, -5.0]
    assert rmse(values, values) == pytest.approx(0.0)


def test_spearman_detects_monotone_relationships():
    y_true = [-9.0, -8.0, -7.0, -6.0, -5.0]
    assert spearman_rho(y_true, [-9.0, -8.0, -7.0, -6.0, -5.0]) == pytest.approx(1.0)
    assert spearman_rho(y_true, [-5.0, -6.0, -7.0, -8.0, -9.0]) == pytest.approx(-1.0)
    # Monotone but non-linear: Spearman is 1, Pearson would not be.
    assert spearman_rho(y_true, [1.0, 2.0, 4.0, 8.0, 16.0]) == pytest.approx(1.0)


def test_spearman_is_nan_for_constant_input():
    """Undefined, not zero: reporting 0.0 would bias bootstrap intervals."""
    assert np.isnan(spearman_rho([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))


def test_enrichment_factor_perfect_and_random():
    """EF caps at 1/fraction and sits at ~1 for an uninformative predictor."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(-8.0, 1.5, size=1000)

    # Perfect ranking.
    assert enrichment_factor(y_true, y_true, 0.01) == pytest.approx(100.0)
    # Perfectly inverted ranking finds none of the true top 1%.
    assert enrichment_factor(y_true, -y_true, 0.01) == pytest.approx(0.0)


def test_enrichment_factor_hand_computed():
    """Ten molecules, top 20% = 2; one of the two predicted best is a hit."""
    y_true = [-9.0, -8.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 0.0]
    # Predicted ranks molecule 0 (a true active) and molecule 9 (the worst)
    # as the two best.
    y_pred = [-99.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -98.0]
    # hit rate 1/2 = 0.5, divided by 0.2 -> 2.5
    assert enrichment_factor(y_true, y_pred, 0.2) == pytest.approx(2.5)


def test_enrichment_factor_respects_the_sign_convention():
    """More negative is better; an inverted predictor must score below 1."""
    y_true = np.linspace(-10.0, 0.0, 200)
    good = enrichment_factor(y_true, y_true, 0.05)
    bad = enrichment_factor(y_true, -y_true, 0.05)
    assert good > 1.0 > bad


def test_precision_at_k_hand_computed():
    y_true = [-5.0, -4.0, -3.0, -2.0, -1.0]
    # Predicts the two true best in the top 2, in the wrong internal order.
    y_pred = [-4.0, -5.0, 0.0, 1.0, 2.0]
    assert precision_at_k(y_true, y_pred, k=2) == pytest.approx(1.0)
    # Predicts only one of the true best two.
    assert precision_at_k(y_true, [-5.0, 9.0, -4.0, 1.0, 2.0], k=2) == pytest.approx(0.5)


def test_precision_at_k_clamps_to_pool_size():
    """k larger than the pool degrades gracefully instead of raising."""
    y_true = [-3.0, -2.0, -1.0]
    assert precision_at_k(y_true, y_true, k=100) == pytest.approx(1.0)


def test_top_k_recall_counts_what_was_labelled():
    y_true = [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0]
    # The true top 3 are indices 0, 1, 2.
    assert top_k_recall(y_true, [0, 1, 2], k=3) == pytest.approx(1.0)
    assert top_k_recall(y_true, [0, 5], k=3) == pytest.approx(1 / 3)
    assert top_k_recall(y_true, [4, 5], k=3) == pytest.approx(0.0)


def test_metric_input_validation():
    with pytest.raises(MetricInputError):
        rmse([1.0, 2.0], [1.0])
    with pytest.raises(MetricInputError):
        rmse([], [])
    with pytest.raises(MetricInputError):
        rmse([1.0, np.nan], [1.0, 2.0])


def test_metric_suite_produces_every_configured_metric():
    rng = np.random.default_rng(1)
    y_true = rng.normal(-8.0, 1.0, size=300)
    y_pred = y_true + rng.normal(0.0, 0.4, size=300)

    suite = MetricSuite(ef_fractions=(0.01, 0.05), precision_k=(10, 50))
    results = suite.compute(y_true, y_pred)

    assert set(results) == {
        "rmse",
        "mae",
        "spearman_rho",
        "pearson_r",
        "ef_01",
        "ef_05",
        "precision_at_10",
        "precision_at_50",
    }
    # A good predictor should beat random enrichment.
    assert results["ef_05"] > 1.0
    assert results["spearman_rho"] > 0.8


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    y_true = rng.normal(-8.0, 1.0, size=200)
    y_pred = y_true + rng.normal(0.0, 0.5, size=200)

    estimate = bootstrap_metric(rmse, y_true, y_pred, name="rmse", n_resamples=200, seed=0)
    assert estimate.ci_low <= estimate.value <= estimate.ci_high
    assert estimate.n_valid_resamples == 200
    assert estimate.ci_level == 0.95


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    rng = np.random.default_rng(3)
    y_true = rng.normal(-8.0, 1.0, size=100)
    y_pred = y_true + rng.normal(0.0, 0.5, size=100)

    first = bootstrap_metric(rmse, y_true, y_pred, name="rmse", n_resamples=100, seed=11)
    second = bootstrap_metric(rmse, y_true, y_pred, name="rmse", n_resamples=100, seed=11)
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_seed_variance_band_shapes_and_values():
    curves = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    mean, low, high = seed_variance_band(curves)
    assert mean == pytest.approx([0.15, 0.35, 0.55])
    assert np.all(low <= mean)
    assert np.all(high >= mean)


def test_seed_variance_band_rejects_wrong_shape():
    with pytest.raises(ValueError, match="2D"):
        seed_variance_band([0.1, 0.2, 0.3])
