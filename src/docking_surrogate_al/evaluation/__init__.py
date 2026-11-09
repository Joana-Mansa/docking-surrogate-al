"""Evaluation subpackage: metrics, bootstrap intervals and calibration."""

from .bootstrap import MetricEstimate, bootstrap_metric, bootstrap_suite, seed_variance_band
from .calibration import CalibrationResult, assess_calibration
from .metrics import (
    MetricSuite,
    enrichment_factor,
    mae,
    precision_at_k,
    rmse,
    spearman_rho,
    top_k_recall,
)
from .report import SplitEvaluation, evaluate_split, write_metrics_csv, write_predictions_csv

__all__ = [
    "CalibrationResult",
    "MetricEstimate",
    "MetricSuite",
    "SplitEvaluation",
    "assess_calibration",
    "bootstrap_metric",
    "bootstrap_suite",
    "enrichment_factor",
    "evaluate_split",
    "mae",
    "precision_at_k",
    "rmse",
    "seed_variance_band",
    "spearman_rho",
    "top_k_recall",
    "write_metrics_csv",
    "write_predictions_csv",
]
