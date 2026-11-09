"""Command-line entry points.

Design note
-----------
Five subcommands, one per pipeline stage: ``prepare-data``, ``dock``,
``train``, ``evaluate`` and ``run-al``. Each is independently runnable and
each reads the stage before it from disk, so a long experiment can be stopped,
inspected and resumed at a stage boundary rather than only as a whole.

Common behaviour, applied by every subcommand before it does any work:

1. Load and validate the config, including ``--set`` overrides.
2. Resolve and *assert* the device, so a run that wants CUDA on a machine
   without it fails in the first second rather than after data preparation.
3. Seed everything and open the run manifest.

``train`` trains one model per configured split so that ``evaluate`` can
report scaffold and random side by side; that pairing is not optional, because
the gap between them is the result.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .al.loop import run_experiment
from .config import Config, ConfigError, REPO_ROOT, config_hash, load_config, resolve_path
from .data.chembl import download_molecules, load_local_csv, write_metadata
from .data.dataset import MoleculePool, TargetScaler, make_loader
from .data.featurise import feature_summary
from .data.splits import assert_no_scaffold_overlap, make_split
from .data.standardise import standardise_frame
from .device import assert_device_available
from .evaluation.metrics import MetricSuite
from .evaluation.report import evaluate_split, evaluation_payload, write_metrics_csv, write_predictions_csv
from .hardware import StageTimer, collect_hardware_info
from .logging_setup import setup_logging
from .manifest import RunManifest
from .models.checkpoint import load_checkpoint
from .models.ensemble import load_ensemble, train_ensemble
from .models.gnn import build_model
from .models.train import train_model
from .models.uncertainty import predict_with_uncertainty
from .oracle import DockingCache, dock_molecules, make_oracle
from .plotting import figures
from .seeds import seed_everything

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Shared setup
# --------------------------------------------------------------------------


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``--set key=value`` pairs, with YAML-ish value coercion."""
    import yaml

    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def _bootstrap(args: argparse.Namespace, stage: str) -> tuple[Config, torch.device, RunManifest]:
    """Load config, assert the device, seed, and open the manifest."""
    cfg = load_config(args.config, _parse_overrides(args.set))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, cfg.run_dir / "run.log")

    logger.info("Stage '%s' starting with config %s.", stage, args.config)
    device = assert_device_available(cfg.run.device)
    seed_everything(cfg.run.seed)

    hardware = collect_hardware_info(device)
    manifest = RunManifest(cfg, hardware, stage)
    if hardware.gpu_name:
        logger.info(
            "GPU: %s, %.0f MiB VRAM, capability %s, CUDA %s.",
            hardware.gpu_name,
            hardware.gpu_total_vram_mb or 0.0,
            hardware.gpu_capability,
            hardware.torch_cuda_version,
        )
    return cfg, device, manifest


def _processed_frame(cfg: Config) -> pd.DataFrame:
    path = resolve_path(cfg.data.processed_path)
    if not path.exists():
        raise SystemExit(
            f"No standardised molecules at {path}. Run `dsal prepare-data --config ...` first."
        )
    return pd.read_csv(path)


def _docked_frame(cfg: Config) -> pd.DataFrame:
    path = cfg.run_dir / "docked.csv"
    if not path.exists():
        raise SystemExit(f"No docking results at {path}. Run `dsal dock --config ...` first.")
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Download (or load) molecules and standardise them."""
    cfg, device, manifest = _bootstrap(args, "prepare-data")

    with StageTimer("prepare_data", device) as timer:
        raw_path = resolve_path(cfg.data.raw_path)
        if cfg.data.source == "chembl":
            frame, metadata = download_molecules(cfg.data.chembl)
        elif cfg.data.source == "csv":
            if not cfg.data.csv_path:
                raise SystemExit("data.source is 'csv' but data.csv_path is not set.")
            frame, metadata = load_local_csv(
                resolve_path(cfg.data.csv_path), cfg.data.smiles_column, cfg.data.id_column
            )
        else:
            raise SystemExit(f"unknown data.source {cfg.data.source!r}; expected 'chembl' or 'csv'")

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(raw_path, index=False)
        write_metadata(raw_path.with_suffix(".provenance.json"), metadata)

        standardised, stats = standardise_frame(
            frame,
            smiles_column="smiles",
            id_column="chembl_id",
            strip_salts=cfg.data.strip_salts,
            neutralise=cfg.data.neutralise,
            min_heavy_atoms=cfg.data.min_heavy_atoms,
            max_heavy_atoms=cfg.data.max_heavy_atoms,
        )
        processed_path = resolve_path(cfg.data.processed_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        standardised.to_csv(processed_path, index=False)
        timer.add_samples(len(standardised))

    manifest.record("data_source", metadata)
    manifest.record("standardisation", stats.to_dict())
    manifest.record("featuriser", feature_summary())
    manifest.record_stage(timer.summary())

    logger.info(
        "prepare-data complete: %d molecules at %s (%.1fs).",
        len(standardised),
        processed_path,
        timer.elapsed_s,
    )
    return 0


def cmd_dock(args: argparse.Namespace) -> int:
    """Dock the standardised molecules to build the ground-truth pool."""
    cfg, device, manifest = _bootstrap(args, "dock")
    frame = _processed_frame(cfg)
    smiles = frame["canonical_smiles"].tolist()

    oracle = make_oracle(cfg.oracle, cfg.target)
    preparation = oracle.prepare()
    manifest.record("oracle_preparation", preparation)
    logger.info("Oracle: %s %s.", oracle.name, oracle.version())

    cache_path = resolve_path(cfg.oracle.cache_path)
    with DockingCache(cache_path, oracle.settings_fingerprint()) as cache:
        with StageTimer("dock", device) as timer:
            stats = dock_molecules(
                oracle,
                smiles,
                cache,
                n_workers=cfg.oracle.n_workers,
                retry_failures=args.retry_failures,
            )
            timer.add_samples(stats.docked)

        scored = cache.successful()
        manifest.record("docking", stats.to_dict())
        manifest.record("docking_cache", cache.summary())

    docked = frame[frame["canonical_smiles"].isin(scored)].copy()
    docked["score"] = docked["canonical_smiles"].map(scored)
    docked["oracle_backend"] = oracle.name
    docked["oracle_version"] = oracle.version()
    out_path = cfg.run_dir / "docked.csv"
    docked.to_csv(out_path, index=False)

    if len(docked):
        figures.plot_score_distribution(
            docked["score"].to_numpy(),
            resolve_path("figures") / f"{cfg.run.name}_score_distribution.png",
        )

    manifest.record_stage(timer.summary())
    failed = len(smiles) - len(docked)
    logger.info(
        "dock complete: %d scored, %d failed, written to %s.", len(docked), failed, out_path
    )
    if failed:
        logger.warning(
            "%d molecule(s) have no score. Failure reasons are recorded per molecule in %s.",
            failed,
            cache_path,
        )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train one model per configured split."""
    cfg, device, manifest = _bootstrap(args, "train")
    frame = _docked_frame(cfg)
    pool = MoleculePool.from_frame(frame)

    split_summaries: dict[str, Any] = {}
    for kind in cfg.evaluation.splits:
        logger.info("=== Training on %s split ===", kind)
        split = make_split(
            kind,
            pool.smiles,
            cfg.evaluation.test_frac,
            cfg.evaluation.val_frac,
            cfg.run.seed,
        )
        if kind == "scaffold":
            assert_no_scaffold_overlap(pool.smiles, split)

        scaler = TargetScaler.fit(pool.scores[split.train])
        train_loader = make_loader(
            pool.subset(split.train, scaler),
            cfg.train,
            shuffle=True,
            seed=cfg.run.seed,
            device_type=device.type,
        )
        val_loader = make_loader(
            pool.subset(split.val, scaler),
            cfg.train,
            shuffle=False,
            seed=cfg.run.seed,
            device_type=device.type,
        )

        checkpoint_dir = cfg.run_dir / "train" / kind
        if cfg.uncertainty.method == "deep_ensemble":
            _, outcomes = train_ensemble(
                cfg,
                device,
                train_loader,
                val_loader,
                scaler,
                checkpoint_dir,
                seed=cfg.run.seed,
                resume=not args.no_resume,
                config_hash=config_hash(cfg),
            )
            best_val = float(np.mean([o.state.best_val_loss for o in outcomes]))
            history = outcomes[0].state.history
            timing = outcomes[0].timing
            amp = outcomes[0].amp_enabled
        else:
            outcome = train_model(
                cfg,
                device,
                train_loader,
                val_loader,
                scaler,
                checkpoint_dir,
                seed=cfg.run.seed,
                resume=not args.no_resume,
                config_hash=config_hash(cfg),
                model=build_model(cfg.model),
            )
            best_val = outcome.state.best_val_loss
            history = outcome.state.history
            timing = outcome.timing
            amp = outcome.amp_enabled

        np.savez(
            checkpoint_dir / "split_indices.npz",
            train=split.train,
            val=split.val,
            test=split.test,
        )
        if history:
            figures.plot_training_curve(
                history, resolve_path("figures") / f"{cfg.run.name}_{kind}_training_curve.png"
            )

        split_summaries[kind] = {
            "sizes": split.sizes(),
            "best_val_loss": best_val,
            "amp_enabled": amp,
            "timing": timing,
        }
        manifest.record_stage({**timing, "stage": f"train_{kind}"})

    manifest.record("training", split_summaries)
    logger.info("train complete for splits: %s.", ", ".join(cfg.evaluation.splits))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate trained models on both splits, side by side."""
    cfg, device, manifest = _bootstrap(args, "evaluate")
    frame = _docked_frame(cfg)
    pool = MoleculePool.from_frame(frame)
    suite = MetricSuite(cfg.evaluation.ef_fractions, cfg.evaluation.precision_k)

    evaluations = []
    calibrations = {}

    with StageTimer("evaluate", device) as timer:
        for kind in cfg.evaluation.splits:
            checkpoint_dir = cfg.run_dir / "train" / kind
            indices_path = checkpoint_dir / "split_indices.npz"
            if not indices_path.exists():
                raise SystemExit(
                    f"No trained model for the {kind} split at {checkpoint_dir}. "
                    "Run `dsal train --config ...` first."
                )
            split_indices = np.load(indices_path)
            test_indices = split_indices["test"]

            loader = make_loader(
                pool.subset(test_indices),
                cfg.train,
                shuffle=False,
                seed=cfg.run.seed,
                device_type=device.type,
            )

            if cfg.uncertainty.method == "deep_ensemble":
                members, scaler = load_ensemble(cfg, checkpoint_dir, device)
                prediction = predict_with_uncertainty(
                    members, loader, device, scaler, method="deep_ensemble"
                )
            else:
                net = build_model(cfg.model).to(device)
                _, scaler = load_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=net,
                    architecture=net.architecture_summary(),
                    restore_rng=False,
                    map_location=device,
                )
                prediction = predict_with_uncertainty(
                    net.to(device),
                    loader,
                    device,
                    scaler,
                    method="mc_dropout",
                    n_passes=cfg.uncertainty.mc_passes,
                    chunk=cfg.uncertainty.mc_chunk,
                )

            # pool_id values are positions in the pool; realign the truth.
            y_true = pool.scores[prediction.pool_id]
            timer.add_samples(len(y_true))

            evaluation = evaluate_split(
                y_true,
                prediction.mean,
                prediction.std,
                cfg.evaluation,
                split=kind,
                seed=cfg.run.seed,
                uncertainty_method=cfg.uncertainty.method,
            )
            evaluations.append(evaluation)
            calibrations[kind] = evaluation.calibration

            write_predictions_csv(
                cfg.run_dir / f"predictions_{kind}.csv",
                smiles=[pool.smiles[i] for i in prediction.pool_id],
                y_true=y_true,
                mean=prediction.mean,
                sigma=prediction.std,
                split=kind,
                seed=cfg.run.seed,
            )
            figures.plot_parity(
                y_true,
                prediction.mean,
                resolve_path("figures") / f"{cfg.run.name}_{kind}_parity.png",
                split=kind,
                rmse_value=evaluation.estimates["rmse"].value,
            )

    metrics_frame = write_metrics_csv(evaluations, cfg.run_dir / "metrics.csv")
    figures.plot_reliability(
        calibrations, resolve_path("figures") / f"{cfg.run.name}_reliability.png"
    )
    if len(evaluations) > 1:
        figures.plot_split_comparison(
            metrics_frame, resolve_path("figures") / f"{cfg.run.name}_split_comparison.png"
        )

    manifest.record("evaluation", evaluation_payload(evaluations, suite))
    manifest.record_stage(timer.summary())

    print("\n=== Evaluation summary ===")
    for evaluation in evaluations:
        print(evaluation.summary_line())
    return 0


def cmd_run_al(args: argparse.Namespace) -> int:
    """Run the active-learning experiment across strategies and seeds."""
    cfg, device, manifest = _bootstrap(args, "run-al")
    frame = _docked_frame(cfg)
    pool = MoleculePool.from_frame(frame)

    with StageTimer("run_al", device) as timer:
        results = run_experiment(
            cfg,
            device,
            pool,
            cfg.run_dir / "al",
            config_hash=config_hash(cfg),
            resume=not args.no_resume,
        )
        timer.add_samples(len(results))

    out_path = cfg.run_dir / "active_learning.csv"
    results.to_csv(out_path, index=False)
    figures.plot_active_learning(
        results,
        resolve_path("figures") / f"{cfg.run.name}_active_learning.png",
        top_k=cfg.al.top_k,
    )

    final = (
        results[results["round_index"] == results["round_index"].max()]
        .groupby("strategy")["top_k_recall"]
        .agg(["mean", "std"])
    )
    manifest.record(
        "active_learning",
        {
            "rounds": cfg.al.rounds,
            "seeds": list(cfg.al.seeds),
            "strategies": list(cfg.al.strategies),
            "retrospective": True,
            "final_top_k_recall": json.loads(final.to_json(orient="index")),
        },
    )
    manifest.record_stage(timer.summary())

    print(f"\n=== Final top-{cfg.al.top_k} recall (mean +- SD over seeds) ===")
    print(final.to_string())
    print(f"\nResults: {out_path}")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="dsal",
        description=(
            "GNN surrogate for molecular docking scores with uncertainty-guided "
            "active learning. Docking scores are not experimental binding "
            "affinities; see the README."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--config",
            type=Path,
            default=REPO_ROOT / "configs" / "default.yaml",
            help="path to the YAML config",
        )
        sub.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set train.epochs=2",
        )
        sub.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="logging verbosity",
        )

    prepare = subparsers.add_parser("prepare-data", help="download and standardise molecules")
    add_common(prepare)
    prepare.set_defaults(func=cmd_prepare_data)

    dock = subparsers.add_parser("dock", help="dock molecules to build the ground-truth pool")
    add_common(dock)
    dock.add_argument(
        "--retry-failures",
        action="store_true",
        help="re-attempt molecules whose cached docking result failed",
    )
    dock.set_defaults(func=cmd_dock)

    train = subparsers.add_parser("train", help="train the surrogate on each split")
    add_common(train)
    train.add_argument(
        "--no-resume", action="store_true", help="ignore existing checkpoints and start fresh"
    )
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate on both splits side by side")
    add_common(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    run_al = subparsers.add_parser("run-al", help="run the active-learning experiment")
    add_common(run_al)
    run_al.add_argument(
        "--no-resume", action="store_true", help="ignore saved campaign state and start fresh"
    )
    run_al.set_defaults(func=cmd_run_al)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        # Checkpoints and the docking cache are written incrementally, so an
        # interrupt here costs at most the work since the last one.
        print("\nInterrupted. Rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
