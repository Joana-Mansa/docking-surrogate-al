"""Assembling evaluation results into CSV and JSON.

Design note
-----------
Results are written as a *long* CSV (one row per split x metric) rather than a
wide one. Long form survives adding a metric or a split without changing the
schema, and it is what plotting and any later meta-analysis want to consume.
Every row repeats the seed and split, so no row's meaning depends on knowing
which file it came from.

Both splits are always evaluated and written together. The gap between them is
the result that justifies the evaluation section existing, so the code makes it
awkward to report one without the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import EvalConfig
from .bootstrap import MetricEstimate, bootstrap_metric
from .calibration import CalibrationResult, assess_calibration
from .metrics import (
    MetricSuite,
    enrichment_factor,
    mae,
    pearson_r,
    precision_at_k,
    rmse,
    spearman_rho,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitEvaluation:
    """Everything measured on one split."""

    split: str
    seed: int
    n_test: int
    estimates: dict[str, MetricEstimate]
    calibration: CalibrationResult
    uncertainty_method: str

    def to_rows(self) -> list[dict[str, Any]]:
        """Return long-form CSV rows for this split."""
        rows = [
            {
                "split": self.split,
                "seed": self.seed,
                "n_test": self.n_test,
                "uncertainty_method": self.uncertainty_method,
                **estimate.to_dict(),
            }
            for estimate in self.estimates.values()
        ]
        # Calibration summaries join the same table so a reader does not have
        # to open two files to see whether the error bars mean anything.
        for name, value in (
            ("expected_calibration_error", self.calibration.expected_calibration_error),
            ("miscalibration_area", self.calibration.miscalibration_area),
            ("sigma_error_spearman", self.calibration.sigma_error_spearman),
            ("mean_sigma", self.calibration.mean_sigma),
        ):
            rows.append(
                {
                    "split": self.split,
                    "seed": self.seed,
                    "n_test": self.n_test,
                    "uncertainty_method": self.uncertainty_method,
                    "metric": name,
                    "value": value,
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "ci_level": float("nan"),
                    "n_resamples": 0,
                    "n_valid_resamples": 0,
                }
            )
        return rows

    def summary_line(self) -> str:
        """One-line human summary for the log."""
        parts = [f"[{self.split}] n={self.n_test}"]
        # Enrichment and precision keys depend on the configured fractions and
        # k values, so they are discovered rather than hardcoded.
        keys = ["rmse", "spearman_rho"]
        keys += sorted(k for k in self.estimates if k.startswith("ef_"))
        keys += sorted(k for k in self.estimates if k.startswith("precision_at_"))
        for key in keys:
            if key in self.estimates:
                parts.append(f"{key}={self.estimates[key].format()}")
        parts.append(f"ECE={self.calibration.expected_calibration_error:.3f}")
        return "  ".join(parts)


def _metric_functions(cfg: EvalConfig) -> dict[str, Any]:
    """Build the metric callables, closing over the configured parameters."""
    functions: dict[str, Any] = {
        "rmse": rmse,
        "mae": mae,
        "spearman_rho": spearman_rho,
        "pearson_r": pearson_r,
    }
    for fraction in cfg.ef_fractions:
        key = f"ef_{fraction:g}".replace("0.", "")
        functions[key] = (
            lambda t, p, _f=fraction: enrichment_factor(t, p, _f)  # bind by value
        )
    for k in cfg.precision_k:
        functions[f"precision_at_{k}"] = lambda t, p, _k=k: precision_at_k(t, p, _k)
    return functions


def evaluate_split(
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    cfg: EvalConfig,
    *,
    split: str,
    seed: int,
    uncertainty_method: str,
) -> SplitEvaluation:
    """Compute bootstrapped metrics and calibration for one split."""
    estimates = {
        name: bootstrap_metric(
            fn,
            y_true,
            mean,
            name=name,
            n_resamples=cfg.bootstrap_samples,
            ci_level=cfg.bootstrap_ci,
            seed=seed,
        )
        for name, fn in _metric_functions(cfg).items()
    }
    calibration = assess_calibration(y_true, mean, sigma, n_bins=cfg.calibration_bins)

    evaluation = SplitEvaluation(
        split=split,
        seed=seed,
        n_test=int(np.asarray(y_true).size),
        estimates=estimates,
        calibration=calibration,
        uncertainty_method=uncertainty_method,
    )
    logger.info("%s", evaluation.summary_line())
    return evaluation


def write_metrics_csv(evaluations: list[SplitEvaluation], path: Path) -> pd.DataFrame:
    """Write the long-form metrics CSV and return it."""
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        rows.extend(evaluation.to_rows())
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote %d metric rows to %s.", len(frame), path)
    return frame


def write_predictions_csv(
    path: Path,
    *,
    smiles: list[str],
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    split: str,
    seed: int,
) -> None:
    """Write per-molecule predictions, so results can be re-analysed."""
    frame = pd.DataFrame(
        {
            "canonical_smiles": smiles,
            "true_score": np.asarray(y_true).ravel(),
            "predicted_score": np.asarray(mean).ravel(),
            "predicted_sigma": np.asarray(sigma).ravel(),
            "split": split,
            "seed": seed,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote %d predictions to %s.", len(frame), path)


def evaluation_payload(evaluations: list[SplitEvaluation], suite: MetricSuite) -> dict[str, Any]:
    """Return a JSON-ready summary for the run manifest."""
    return {
        "metric_definitions": dict(suite.describe()),
        "splits": {
            evaluation.split: {
                "n_test": evaluation.n_test,
                "seed": evaluation.seed,
                "uncertainty_method": evaluation.uncertainty_method,
                "metrics": {
                    name: estimate.to_dict() for name, estimate in evaluation.estimates.items()
                },
                "calibration": evaluation.calibration.to_dict(),
            }
            for evaluation in evaluations
        },
    }
