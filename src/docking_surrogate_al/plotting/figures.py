"""Figures.

Design note
-----------
All figures are saved at 300 dpi to ``figures/`` and use a non-interactive
backend so they render identically in CI and over SSH.

Plotting choices follow from what the figures have to show honestly:

* The active-learning figure draws the **random baseline in the same style and
  weight** as the two active strategies, not as a faint dashed afterthought.
  It is the comparison the plot exists to make.
* Bands are **seed spread (mean +- 1 SD across seeds)**, labelled as such in
  every caption and legend, and never mixed with bootstrap intervals in the
  same panel, because the two answer different questions.
* The parity plot draws the y = x line and fixes both axes to the same range.
  Without that, a visually convincing diagonal cloud can hide a systematic
  offset of several kcal/mol.
* The reliability plot draws the diagonal and shades above/below it, since
  "which side of the diagonal" (underconfident vs overconfident) is the
  actionable part and is easy to misread from the curve alone.

No figure has a title baked in beyond a short descriptive one; captions
belong in the README and the paper, where they can be precise about what the
numbers do and do not support.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..evaluation.bootstrap import seed_variance_band  # noqa: E402
from ..evaluation.calibration import CalibrationResult  # noqa: E402

logger = logging.getLogger(__name__)

DPI = 300
STRATEGY_COLOURS = {
    "uncertainty": "#1f78b4",
    "greedy": "#e31a1c",
    "random": "#33a02c",
}
SPLIT_COLOURS = {"scaffold": "#6a3d9a", "random": "#ff7f00"}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure %s.", path)
    return path


def plot_active_learning(frame: pd.DataFrame, path: Path, top_k: int = 100) -> Path:
    """Top-k recall against oracle calls, with seed-spread bands.

    Args:
        frame: Long-form AL results with strategy, seed, oracle_calls and
            top_k_recall columns.
        path: Output path.
        top_k: The k used, for the axis label.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for strategy, group in frame.groupby("strategy"):
        pivot = group.pivot_table(
            index="seed", columns="round_index", values="top_k_recall", aggfunc="mean"
        ).sort_index(axis=1)
        calls = (
            group.pivot_table(
                index="seed", columns="round_index", values="oracle_calls", aggfunc="mean"
            )
            .sort_index(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
        mean, low, high = seed_variance_band(pivot.to_numpy())
        colour = STRATEGY_COLOURS.get(str(strategy), None)
        ax.plot(calls, mean, marker="o", markersize=4, label=str(strategy), color=colour, lw=2)
        ax.fill_between(calls, low, high, alpha=0.18, color=colour, linewidth=0)

    ax.set_xlabel("Oracle calls (molecules docked)")
    ax.set_ylabel(f"Top-{top_k} recall")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3, linewidth=0.6)
    n_seeds = frame["seed"].nunique()
    ax.legend(title=f"Strategy (band: mean $\\pm$ 1 SD over {n_seeds} seeds)", frameon=False)
    ax.set_title(f"Top-{top_k} recall vs oracle calls", fontsize=11)
    return _save(fig, path)


def plot_parity(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path, *, split: str, rmse_value: float
) -> Path:
    """Predicted vs true docking score, with the y = x line."""
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    colour = SPLIT_COLOURS.get(split, "#444444")
    ax.scatter(y_true, y_pred, s=12, alpha=0.45, color=colour, edgecolors="none")

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    limits = (lo - pad, hi + pad)
    ax.plot(limits, limits, color="black", lw=1.0, ls="--", label="y = x")

    # Identical limits on both axes: a parity plot with different ranges can
    # make a systematic offset look like a tight diagonal.
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True docking score (kcal/mol)")
    ax.set_ylabel("Predicted docking score (kcal/mol)")
    ax.set_title(f"{split} split, RMSE = {rmse_value:.2f} kcal/mol", fontsize=11)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)


def plot_reliability(results: dict[str, CalibrationResult], path: Path) -> Path:
    """Reliability (coverage) curves for every split, plus the diagonal."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.6))

    ax.plot([0, 1], [0, 1], color="black", lw=1.0, ls="--", label="perfect calibration")
    for split, result in results.items():
        ax.plot(
            result.nominal_levels,
            result.observed_coverage,
            marker="o",
            markersize=4,
            lw=2,
            color=SPLIT_COLOURS.get(split),
            label=f"{split} (ECE = {result.expected_calibration_error:.3f})",
        )
    ax.fill_between([0, 1], [0, 1], [1, 1], color="grey", alpha=0.07, linewidth=0)
    ax.text(0.62, 0.20, "overconfident", fontsize=8, color="grey")
    ax.text(0.16, 0.86, "underconfident", fontsize=8, color="grey")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Interval coverage", fontsize=11)

    # Second panel: does sigma actually track error?
    for split, result in results.items():
        mask = np.isfinite(result.bin_mean_sigma) & np.isfinite(result.bin_rmse)
        ax2.plot(
            result.bin_mean_sigma[mask],
            result.bin_rmse[mask],
            marker="s",
            markersize=4,
            lw=2,
            color=SPLIT_COLOURS.get(split),
            label=f"{split} (rank corr = {result.sigma_error_spearman:.2f})",
        )
    all_values = np.concatenate(
        [
            np.concatenate([r.bin_mean_sigma[np.isfinite(r.bin_mean_sigma)], r.bin_rmse[np.isfinite(r.bin_rmse)]])
            for r in results.values()
        ]
    )
    if all_values.size:
        hi = float(np.nanmax(all_values)) * 1.05
        ax2.plot([0, hi], [0, hi], color="black", lw=1.0, ls="--", label="y = x")
    ax2.set_xlabel("Predicted $\\sigma$ per bin (kcal/mol)")
    ax2.set_ylabel("Empirical RMSE per bin (kcal/mol)")
    ax2.grid(alpha=0.3, linewidth=0.6)
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("Error vs uncertainty (quantile bins)", fontsize=11)

    fig.tight_layout()
    return _save(fig, path)


def plot_split_comparison(metrics: pd.DataFrame, path: Path) -> Path:
    """Side-by-side metric comparison between scaffold and random splits."""
    # Enrichment and precision metric names depend on the configured
    # fractions and k values, so they are discovered from the frame.
    available = set(metrics["metric"])
    interesting = ["rmse", "spearman_rho"]
    interesting += sorted(m for m in available if m.startswith("ef_"))
    interesting += sorted(m for m in available if m.startswith("precision_at_"))

    subset = metrics[metrics["metric"].isin(interesting)].copy()
    if subset.empty:
        raise ValueError("no comparable metrics found in the results frame")

    order = [m for m in interesting if m in available]
    splits = sorted(set(subset["split"]))
    width = 0.8 / max(len(splits), 1)

    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 3.8))
    if len(order) == 1:
        axes = [axes]

    for ax, metric in zip(axes, order, strict=True):
        rows = subset[subset["metric"] == metric]
        for i, split in enumerate(splits):
            row = rows[rows["split"] == split]
            if row.empty:
                continue
            value = float(row["value"].iloc[0])
            low = float(row["ci_low"].iloc[0])
            high = float(row["ci_high"].iloc[0])
            # Error bars are bootstrap CIs here, not seed spread.
            err = np.array([[max(0.0, value - low)], [max(0.0, high - value)]])
            ax.bar(
                i * width,
                value,
                width=width * 0.9,
                color=SPLIT_COLOURS.get(split),
                label=split,
                yerr=err if np.all(np.isfinite(err)) else None,
                capsize=4,
            )
        ax.set_xticks([i * width for i in range(len(splits))])
        ax.set_xticklabels(splits, fontsize=8)
        ax.set_title(metric, fontsize=10)
        ax.grid(alpha=0.3, axis="y", linewidth=0.6)

    fig.suptitle("Scaffold vs random split (error bars: bootstrap 95% CI)", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_curve(history: list[dict[str, float]], path: Path) -> Path:
    """Training and validation loss per epoch."""
    if not history:
        raise ValueError("no training history to plot")
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(frame["epoch"], frame["train_loss"], label="train", lw=2)
    ax.plot(frame["epoch"], frame["val_loss"], label="validation", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised target space)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_title("Training curve", fontsize=11)
    return _save(fig, path)


def plot_score_distribution(scores: np.ndarray, path: Path) -> Path:
    """Distribution of oracle docking scores over the pool."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(scores, bins=50, color="#4c72b0", alpha=0.85)
    ax.axvline(
        float(np.percentile(scores, 1)),
        color="#e31a1c",
        ls="--",
        lw=1.5,
        label="top 1% threshold",
    )
    ax.set_xlabel("Docking score (kcal/mol, more negative is better)")
    ax.set_ylabel("Molecules")
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_title("Oracle score distribution", fontsize=11)
    return _save(fig, path)
