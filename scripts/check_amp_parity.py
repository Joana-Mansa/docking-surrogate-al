"""Check that mixed precision does not change results beyond seed variance.

Why this exists
---------------
``train.amp`` defaults to on, which is a claim that fp16 autocast is free: the
same answer, faster. That claim needs checking, not asserting, because AMP
genuinely can change results (loss scaling interacts with gradient clipping,
and fp16 accumulation differs from fp32).

The check is a comparison of two *distributions*, not two numbers. Comparing a
single AMP run to a single FP32 run tells you nothing, because two FP32 runs
with different seeds already differ. So this script trains the same config
with AMP on and off across N seeds each, and asks whether the AMP-vs-FP32
difference in mean test RMSE is small relative to the seed-to-seed spread.

Pass criterion: |mean(AMP) - mean(FP32)| < seed standard deviation of the
FP32 runs. That is a deliberately loose bar. It establishes "AMP does not
change results beyond seed variance", which is exactly the claim the README
makes; it does not establish bit-identity, which would be false.

Requires a CUDA GPU, since AMP is disabled on CPU. On a CPU-only machine it
exits with a clear message rather than silently comparing two identical runs.

Usage:
    python scripts/check_amp_parity.py --config configs/default.yaml --seeds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from docking_surrogate_al.config import REPO_ROOT, config_hash, load_config
from docking_surrogate_al.data.dataset import MoleculePool, TargetScaler, make_loader
from docking_surrogate_al.data.splits import make_split
from docking_surrogate_al.device import assert_device_available
from docking_surrogate_al.evaluation.metrics import rmse, spearman_rho
from docking_surrogate_al.logging_setup import setup_logging
from docking_surrogate_al.models.gnn import build_model
from docking_surrogate_al.models.train import predict_mean, train_model
from docking_surrogate_al.seeds import seed_everything


def run_once(cfg, device, pool, split, *, seed: int, amp: bool, workspace: Path) -> dict[str, float]:
    """Train once and return test metrics."""
    seed_everything(seed)
    scaler = TargetScaler.fit(pool.scores[split.train])
    train_loader = make_loader(
        pool.subset(split.train, scaler), cfg.train, shuffle=True, seed=seed, device_type=device.type
    )
    val_loader = make_loader(
        pool.subset(split.val, scaler), cfg.train, shuffle=False, seed=seed, device_type=device.type
    )
    test_loader = make_loader(
        pool.subset(split.test), cfg.train, shuffle=False, seed=seed, device_type=device.type
    )

    import dataclasses

    cfg_variant = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, amp=amp))
    outcome = train_model(
        cfg_variant,
        device,
        train_loader,
        val_loader,
        scaler,
        workspace / f"amp{int(amp)}_seed{seed}",
        seed=seed,
        resume=False,
        config_hash=config_hash(cfg_variant),
        model=build_model(cfg.model),
    )

    predictions, pool_ids = predict_mean(outcome.model, test_loader, device)
    order = np.argsort(pool_ids)
    y_pred = scaler.inverse(predictions[order])
    y_true = pool.scores[pool_ids[order]]

    return {
        "seed": seed,
        "amp": amp,
        "amp_actually_enabled": outcome.amp_enabled,
        "test_rmse": rmse(y_true, y_pred),
        "test_spearman": spearman_rho(y_true, y_pred),
        "best_val_loss": outcome.state.best_val_loss,
        "wall_clock_s": outcome.timing["wall_clock_s"],
        "peak_vram_mb": outcome.timing.get("peak_allocated_mb"),
    }


def main() -> int:
    """Run the parity check and print a verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--seeds", type=int, default=3, help="seeds per arm")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "runs" / "amp_parity", help="output directory"
    )
    args = parser.parse_args()

    setup_logging("INFO")
    cfg = load_config(args.config)
    if args.epochs is not None:
        import dataclasses

        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, epochs=args.epochs))

    if not torch.cuda.is_available():
        print(
            "This check needs a CUDA GPU: AMP is disabled on CPU, so both arms "
            "would be identical and the comparison would be meaningless."
        )
        return 1

    device = assert_device_available("cuda")
    docked = cfg.run_dir / "docked.csv"
    if not docked.exists():
        print(f"No docking results at {docked}. Run prepare-data and dock first.")
        return 1

    pool = MoleculePool.from_frame(pd.read_csv(docked))
    split = make_split(
        "scaffold", pool.smiles, cfg.evaluation.test_frac, cfg.evaluation.val_frac, cfg.run.seed
    )

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        run_once(cfg, device, pool, split, seed=seed, amp=amp, workspace=args.out)
        for amp in (False, True)
        for seed in range(args.seeds)
    ]

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "amp_parity.csv", index=False)

    fp32 = frame[~frame["amp"]]["test_rmse"].to_numpy()
    amp16 = frame[frame["amp"]]["test_rmse"].to_numpy()
    seed_sd = float(np.std(fp32, ddof=1)) if fp32.size > 1 else float("nan")
    gap = float(abs(amp16.mean() - fp32.mean()))
    passed = bool(gap < seed_sd)

    verdict = {
        "config": str(args.config),
        "seeds_per_arm": args.seeds,
        "fp32_mean_rmse": float(fp32.mean()),
        "amp_mean_rmse": float(amp16.mean()),
        "absolute_gap": gap,
        "fp32_seed_sd": seed_sd,
        "passed": passed,
        "criterion": "|mean(AMP) - mean(FP32)| < seed SD of the FP32 arm",
        "fp32_mean_wall_clock_s": float(frame[~frame["amp"]]["wall_clock_s"].mean()),
        "amp_mean_wall_clock_s": float(frame[frame["amp"]]["wall_clock_s"].mean()),
        "fp32_mean_peak_vram_mb": float(frame[~frame["amp"]]["peak_vram_mb"].mean()),
        "amp_mean_peak_vram_mb": float(frame[frame["amp"]]["peak_vram_mb"].mean()),
    }
    (args.out / "amp_parity.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print("\n=== AMP parity check ===")
    for key, value in verdict.items():
        print(f"  {key}: {value}")
    print(f"\nFull results: {args.out / 'amp_parity.csv'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
