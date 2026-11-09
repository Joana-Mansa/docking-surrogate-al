"""Tests for uncertainty estimation, calibration and acquisition strategies."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from docking_surrogate_al.config import ModelConfig, TrainConfig
from docking_surrogate_al.data.dataset import MoleculePool, TargetScaler, make_loader
from docking_surrogate_al.evaluation.calibration import (
    assess_calibration,
    coverage_curve,
    error_vs_uncertainty,
)
from docking_surrogate_al.models.gnn import build_model
from docking_surrogate_al.models.uncertainty import mc_dropout_predict
from docking_surrogate_al.al.strategies import (
    UnknownStrategyError,
    initial_selection,
    select_batch,
)

SMILES = [
    "CCOc1ccccc1C(=O)O",
    "NC(=O)c1ccccc1",
    "Cc1c[nH]c2ccccc12",
    "c1ccc2ncccc2c1",
    "c1ccc(N2CCNCC2)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCNC(=O)c1ccccc1",
    "OCc1c[nH]c2ccccc12",
]
SCORES = [-7.1, -6.4, -8.2, -7.8, -6.9, -7.3, -6.1, -8.0]


# --- MC-dropout -----------------------------------------------------------


def test_mc_dropout_produces_a_nonzero_spread():
    """Replicated passes must actually differ, or the uncertainty is fake."""
    pool = MoleculePool(SMILES, SCORES)
    scaler = TargetScaler.fit(pool.scores)
    loader = make_loader(
        pool.subset(range(len(pool)), scaler),
        TrainConfig(batch_size=4, num_workers=0, pin_memory=False),
        shuffle=False,
        seed=0,
        device_type="cpu",
    )
    model = build_model(ModelConfig(hidden_dim=16, num_layers=2, dropout=0.3))

    prediction = mc_dropout_predict(
        model, loader, torch.device("cpu"), scaler, n_passes=8, chunk=4
    )

    assert prediction.mean.shape == (len(pool),)
    assert prediction.std.shape == (len(pool),)
    assert np.all(prediction.std > 0), "dropout produced identical passes"
    assert prediction.n_samples == 8
    assert prediction.method == "mc_dropout"


def test_mc_dropout_returns_pool_indices_in_order():
    """Predictions are realigned to pool order, not loader order."""
    pool = MoleculePool(SMILES, SCORES)
    scaler = TargetScaler.fit(pool.scores)
    loader = make_loader(
        pool.subset(range(len(pool)), scaler),
        TrainConfig(batch_size=3, num_workers=0, pin_memory=False),
        shuffle=False,
        seed=0,
        device_type="cpu",
    )
    model = build_model(ModelConfig(hidden_dim=16, num_layers=2, dropout=0.2))
    prediction = mc_dropout_predict(
        model, loader, torch.device("cpu"), scaler, n_passes=4, chunk=2
    )
    assert np.array_equal(prediction.pool_id, np.arange(len(pool)))


def test_mc_dropout_chunking_does_not_change_the_number_of_samples():
    """chunk is a memory knob, not a statistical one."""
    pool = MoleculePool(SMILES, SCORES)
    scaler = TargetScaler.fit(pool.scores)
    model = build_model(ModelConfig(hidden_dim=16, num_layers=2, dropout=0.25))

    def run(chunk: int):
        loader = make_loader(
            pool.subset(range(len(pool)), scaler),
            TrainConfig(batch_size=4, num_workers=0, pin_memory=False),
            shuffle=False,
            seed=0,
            device_type="cpu",
        )
        return mc_dropout_predict(
            model, loader, torch.device("cpu"), scaler, n_passes=12, chunk=chunk
        )

    assert run(3).n_samples == run(12).n_samples == 12


def test_mc_dropout_requires_at_least_two_passes():
    pool = MoleculePool(SMILES, SCORES)
    scaler = TargetScaler.fit(pool.scores)
    loader = make_loader(
        pool.subset(range(len(pool)), scaler),
        TrainConfig(batch_size=4, num_workers=0, pin_memory=False),
        shuffle=False,
        seed=0,
        device_type="cpu",
    )
    with pytest.raises(ValueError, match="at least 2"):
        mc_dropout_predict(
            build_model(ModelConfig(hidden_dim=8, num_layers=1)),
            loader,
            torch.device("cpu"),
            scaler,
            n_passes=1,
        )


# --- Calibration ----------------------------------------------------------


def test_perfectly_calibrated_gaussian_scores_near_zero_ece():
    """Sanity check against synthetic, genuinely calibrated predictions."""
    rng = np.random.default_rng(0)
    n = 20000
    mean = rng.normal(-8.0, 1.0, size=n)
    sigma = np.full(n, 0.5)
    y_true = mean + rng.normal(0.0, 0.5, size=n)

    result = assess_calibration(y_true, mean, sigma, n_bins=10)
    assert result.expected_calibration_error < 0.02
    assert result.miscalibration_area < 0.02


def test_overconfident_predictions_sit_below_the_diagonal():
    """Sigma too small means observed coverage below nominal."""
    rng = np.random.default_rng(1)
    n = 5000
    mean = np.zeros(n)
    y_true = rng.normal(0.0, 1.0, size=n)
    sigma = np.full(n, 0.2)  # four times too small

    nominal, observed = coverage_curve(y_true, mean, sigma, n_levels=10)
    assert np.all(observed <= nominal + 1e-9)
    assert assess_calibration(y_true, mean, sigma).expected_calibration_error > 0.2


def test_underconfident_predictions_sit_above_the_diagonal():
    rng = np.random.default_rng(2)
    n = 5000
    mean = np.zeros(n)
    y_true = rng.normal(0.0, 1.0, size=n)
    sigma = np.full(n, 4.0)  # four times too large

    nominal, observed = coverage_curve(y_true, mean, sigma, n_levels=10)
    assert np.all(observed >= nominal - 1e-9)


def test_sigma_that_tracks_error_gives_positive_rank_correlation():
    """The discriminative question, separate from the coverage question."""
    rng = np.random.default_rng(3)
    n = 2000
    sigma = rng.uniform(0.1, 2.0, size=n)
    mean = np.zeros(n)
    y_true = rng.normal(0.0, sigma)  # error scales with sigma by construction

    result = assess_calibration(y_true, mean, sigma, n_bins=10)
    assert result.sigma_error_spearman > 0.3
    valid = np.isfinite(result.bin_rmse)
    # Error should rise across sigma bins.
    assert result.bin_rmse[valid][-1] > result.bin_rmse[valid][0]


def test_error_vs_uncertainty_uses_quantile_bins():
    """Bins hold comparable counts even when sigma is heavily skewed."""
    rng = np.random.default_rng(4)
    sigma = np.concatenate([rng.uniform(0.1, 0.2, 900), rng.uniform(5.0, 9.0, 100)])
    mean = np.zeros(sigma.size)
    y_true = rng.normal(0.0, sigma)

    _, _, _, counts = error_vs_uncertainty(y_true, mean, sigma, n_bins=5)
    assert counts.sum() == sigma.size
    assert counts.min() > 0


def test_constant_sigma_does_not_crash_calibration():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    result = assess_calibration(y_true, np.zeros(4), np.ones(4), n_bins=4)
    assert np.isnan(result.sigma_error_spearman)


# --- Acquisition strategies ----------------------------------------------


def test_uncertainty_strategy_picks_the_most_uncertain():
    candidates = np.array([10, 11, 12, 13, 14])
    sigma = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    selected = select_batch(
        "uncertainty",
        candidate_indices=candidates,
        predicted_mean=np.zeros(5),
        predicted_sigma=sigma,
        batch_size=2,
        rng=np.random.default_rng(0),
    )
    assert set(selected.tolist()) == {11, 13}


def test_greedy_strategy_picks_the_most_negative_prediction():
    """The sign convention again: best means most negative."""
    candidates = np.array([0, 1, 2, 3])
    mean = np.array([-5.0, -9.0, -1.0, -8.0])
    selected = select_batch(
        "greedy",
        candidate_indices=candidates,
        predicted_mean=mean,
        predicted_sigma=np.ones(4),
        batch_size=2,
        rng=np.random.default_rng(0),
    )
    assert set(selected.tolist()) == {1, 3}


def test_random_strategy_is_seeded_and_stays_in_the_candidate_set():
    candidates = np.arange(100, 200)
    kwargs = dict(
        candidate_indices=candidates,
        predicted_mean=None,
        predicted_sigma=None,
        batch_size=10,
    )
    first = select_batch("random", rng=np.random.default_rng(5), **kwargs)
    again = select_batch("random", rng=np.random.default_rng(5), **kwargs)
    different = select_batch("random", rng=np.random.default_rng(6), **kwargs)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, different)
    assert set(first.tolist()) <= set(candidates.tolist())
    assert len(set(first.tolist())) == 10


def test_selection_never_exceeds_the_candidate_pool():
    candidates = np.array([1, 2, 3])
    selected = select_batch(
        "random",
        candidate_indices=candidates,
        predicted_mean=None,
        predicted_sigma=None,
        batch_size=50,
        rng=np.random.default_rng(0),
    )
    assert sorted(selected.tolist()) == [1, 2, 3]


def test_empty_candidate_set_returns_empty():
    selected = select_batch(
        "uncertainty",
        candidate_indices=[],
        predicted_mean=np.array([]),
        predicted_sigma=np.array([]),
        batch_size=5,
        rng=np.random.default_rng(0),
    )
    assert selected.size == 0


def test_unknown_strategy_raises():
    with pytest.raises(UnknownStrategyError):
        select_batch(
            "thompson",
            candidate_indices=[1, 2],
            predicted_mean=np.zeros(2),
            predicted_sigma=np.ones(2),
            batch_size=1,
            rng=np.random.default_rng(0),
        )


def test_mismatched_prediction_length_raises():
    with pytest.raises(ValueError, match="candidates"):
        select_batch(
            "uncertainty",
            candidate_indices=[1, 2, 3],
            predicted_mean=np.zeros(3),
            predicted_sigma=np.ones(2),
            batch_size=1,
            rng=np.random.default_rng(0),
        )


def test_initial_selection_is_shared_across_strategies_for_a_seed():
    """Every strategy must start from the same molecules for a given seed."""
    a = initial_selection(1000, 50, np.random.default_rng(7))
    b = initial_selection(1000, 50, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert len(set(a.tolist())) == 50


def test_initial_selection_rejects_an_oversized_request():
    with pytest.raises(ValueError, match="exceeds the pool size"):
        initial_selection(10, 50, np.random.default_rng(0))
