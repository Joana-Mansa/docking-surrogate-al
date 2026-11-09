"""The active-learning loop.

Design note
-----------
This is a *retrospective* active-learning experiment. The whole pool is docked
once up front, and a round "calling the oracle" means revealing labels that
were already computed. Nothing is docked during the loop.

That is a deliberate and standard design, and its consequences are worth
stating because they bound what the result means:

* It is what makes 3 strategies x 5 seeds x 10 rounds affordable at all. Done
  prospectively, that is 150 separate docking campaigns.
* Every strategy sees the identical oracle, with no run-to-run docking noise,
  so differences between curves come from acquisition alone.
* It cannot discover molecules outside the pool. A real campaign enumerates
  new chemistry as it learns; this one re-ranks a fixed library. The curves
  therefore measure sample efficiency *within a fixed pool*, which is a
  narrower claim than "active learning finds better molecules".

The reported quantity is top-k recall against number of oracle calls: of the
k best-scoring molecules in the pool, how many has the campaign paid for. A
campaign that has labelled n molecules has made exactly n oracle calls, so the
x-axis is unambiguous and comparable across strategies.

Resumability
------------
State is checkpointed after every round, before and independently of the model
checkpoints that :mod:`docking_surrogate_al.models.train` writes per epoch. A
campaign killed mid-round resumes from the last completed round: the labelled
set, the round index, the RNG state and the history are all restored, so a
resumed campaign selects exactly what an uninterrupted one would have.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.dataset import MoleculePool, TargetScaler, make_loader
from ..evaluation.metrics import spearman_rho, top_k_recall
from ..models.ensemble import train_ensemble
from ..models.gnn import build_model
from ..models.train import train_model
from ..models.uncertainty import UncertainPrediction, predict_with_uncertainty
from ..seeds import seed_everything
from .strategies import initial_selection, select_batch

logger = logging.getLogger(__name__)


@dataclass
class RoundRecord:
    """One completed active-learning round."""

    round_index: int
    strategy: str
    seed: int
    n_labelled: int
    oracle_calls: int
    top_k_recall: float
    best_score_found: float
    mean_score_labelled: float
    pool_spearman: float
    val_loss: float

    def to_dict(self) -> dict[str, Any]:
        """Return a CSV-ready mapping."""
        return asdict(self)


@dataclass
class CampaignState:
    """Resumable state of one (strategy, seed) campaign."""

    strategy: str
    seed: int
    completed_rounds: int = 0
    labelled: list[int] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    rng_state: bytes | None = None


def _state_path(root: Path) -> Path:
    return root / "campaign_state.json"


def save_campaign_state(root: Path, state: CampaignState, rng: np.random.Generator) -> None:
    """Atomically checkpoint campaign state after a completed round."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": state.strategy,
        "seed": state.seed,
        "completed_rounds": state.completed_rounds,
        "labelled": [int(i) for i in state.labelled],
        "history": state.history,
        # The generator state is what makes a resumed campaign select the same
        # molecules an uninterrupted one would have.
        "rng_state_hex": pickle.dumps(rng.bit_generator.state).hex(),
    }
    path = _state_path(root)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def load_campaign_state(root: Path) -> tuple[CampaignState, np.random.Generator] | None:
    """Load campaign state, or None if there is nothing to resume."""
    path = _state_path(root)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError:
        logger.warning("Campaign state at %s is unreadable; starting fresh.", path)
        return None

    state = CampaignState(
        strategy=payload["strategy"],
        seed=payload["seed"],
        completed_rounds=int(payload["completed_rounds"]),
        labelled=[int(i) for i in payload["labelled"]],
        history=list(payload.get("history", [])),
    )
    rng = np.random.default_rng(state.seed)
    if payload.get("rng_state_hex"):
        try:
            rng.bit_generator.state = pickle.loads(bytes.fromhex(payload["rng_state_hex"]))
        except (ValueError, pickle.UnpicklingError) as exc:
            logger.warning("Could not restore campaign RNG state (%s); reseeding.", exc)
    logger.info(
        "Resuming %s campaign (seed %d) after round %d with %d labelled molecules.",
        state.strategy,
        state.seed,
        state.completed_rounds,
        len(state.labelled),
    )
    return state, rng


def run_campaign(
    cfg: Config,
    device: torch.device,
    pool: MoleculePool,
    *,
    strategy: str,
    seed: int,
    run_root: Path,
    config_hash: str = "",
    resume: bool = True,
) -> list[RoundRecord]:
    """Run one (strategy, seed) active-learning campaign.

    Args:
        cfg: Full run config.
        device: Resolved device.
        pool: The fully docked molecule pool.
        strategy: Acquisition strategy name.
        seed: Campaign seed.
        run_root: Directory for this campaign's checkpoints and state.
        config_hash: Recorded in model checkpoints.
        resume: Resume from a saved campaign state if present.

    Returns:
        One :class:`RoundRecord` per completed round, including round 0.
    """
    campaign_root = run_root / f"{strategy}_seed{seed}"
    campaign_root.mkdir(parents=True, exist_ok=True)

    restored = load_campaign_state(campaign_root) if resume else None
    if restored is not None:
        state, rng = restored
    else:
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        state = CampaignState(
            strategy=strategy,
            seed=seed,
            labelled=initial_selection(len(pool), cfg.al.initial_size, rng).tolist(),
        )

    records = [RoundRecord(**record) for record in state.history]
    # Predictions from the previous round, reused for this round's selection.
    # None on a fresh resume, where they are recomputed from the checkpoint.
    prediction: UncertainPrediction | None = None

    for round_index in range(state.completed_rounds, cfg.al.rounds + 1):
        # Round 0 trains on the initial random set without acquiring anything,
        # which is the shared starting point every strategy's curve begins at.
        if round_index > 0:
            if prediction is None:
                logger.info("Recomputing round %d predictions after resume.", round_index - 1)
                prediction = _predict_pool(
                    cfg, device, pool, state, campaign_root, round_index - 1, config_hash, seed
                )
            unlabelled = _unlabelled_indices(len(pool), state.labelled)
            selected = select_batch(
                strategy,
                candidate_indices=unlabelled,
                predicted_mean=prediction.mean[unlabelled],
                predicted_sigma=prediction.std[unlabelled],
                batch_size=cfg.al.batch_size,
                rng=rng,
            )
            state.labelled = sorted(set(state.labelled) | set(int(i) for i in selected))
            logger.info(
                "Round %d (%s, seed %d): acquired %d molecules, %d labelled in total.",
                round_index,
                strategy,
                seed,
                len(selected),
                len(state.labelled),
            )

        record, prediction = _train_and_score(
            cfg, device, pool, state, campaign_root, round_index, config_hash, seed, strategy
        )
        records.append(record)
        state.history = [r.to_dict() for r in records]
        state.completed_rounds = round_index + 1
        save_campaign_state(campaign_root, state, rng)

        if len(state.labelled) >= len(pool):
            logger.info("Pool exhausted after round %d; stopping campaign.", round_index)
            break

    return records


def _unlabelled_indices(pool_size: int, labelled: list[int]) -> np.ndarray:
    mask = np.ones(pool_size, dtype=bool)
    mask[np.asarray(labelled, dtype=np.int64)] = False
    return np.flatnonzero(mask)


def _loaders(
    cfg: Config, pool: MoleculePool, labelled: list[int], seed: int, device: torch.device
) -> tuple[Any, Any, TargetScaler, np.ndarray]:
    """Build train/val loaders from the labelled set.

    The validation split is carved out of the *labelled* molecules, because in
    a real campaign those are the only labels available. Holding out unlabelled
    molecules for validation would use scores the campaign has not paid for.
    """
    labelled_array = np.asarray(labelled, dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(labelled_array)
    n_val = max(1, int(round(len(shuffled) * cfg.evaluation.val_frac)))
    val_indices = np.sort(shuffled[:n_val])
    train_indices = np.sort(shuffled[n_val:])

    scaler = TargetScaler.fit(pool.scores[train_indices])
    train_loader = make_loader(
        pool.subset(train_indices, scaler),
        cfg.train,
        shuffle=True,
        seed=seed,
        device_type=device.type,
    )
    val_loader = make_loader(
        pool.subset(val_indices, scaler),
        cfg.train,
        shuffle=False,
        seed=seed,
        device_type=device.type,
    )
    return train_loader, val_loader, scaler, train_indices


def _train_and_score(
    cfg: Config,
    device: torch.device,
    pool: MoleculePool,
    state: CampaignState,
    campaign_root: Path,
    round_index: int,
    config_hash: str,
    seed: int,
    strategy: str,
) -> tuple[RoundRecord, UncertainPrediction]:
    """Retrain on the labelled set and score the round.

    Returns the round record and the pool-wide predictions, which the caller
    reuses for the next round's acquisition instead of recomputing them.
    """
    train_loader, val_loader, scaler, _ = _loaders(cfg, pool, state.labelled, seed, device)
    round_dir = campaign_root / f"round_{round_index:02d}"

    if cfg.uncertainty.method == "deep_ensemble":
        _, outcomes = train_ensemble(
            cfg,
            device,
            train_loader,
            val_loader,
            scaler,
            round_dir,
            seed=seed,
            epochs=cfg.al.epochs_per_round,
            config_hash=config_hash,
        )
        val_loss = float(np.mean([o.state.best_val_loss for o in outcomes]))
    else:
        outcome = train_model(
            cfg,
            device,
            train_loader,
            val_loader,
            scaler,
            round_dir,
            seed=seed,
            epochs=cfg.al.epochs_per_round,
            config_hash=config_hash,
            model=build_model(cfg.model),
        )
        val_loss = outcome.state.best_val_loss

    prediction = _predict_pool(cfg, device, pool, state, campaign_root, round_index, config_hash, seed)
    labelled_scores = pool.scores[np.asarray(state.labelled, dtype=np.int64)]

    record = RoundRecord(
        round_index=round_index,
        strategy=strategy,
        seed=seed,
        n_labelled=len(state.labelled),
        # One label == one oracle call, by construction.
        oracle_calls=len(state.labelled),
        top_k_recall=top_k_recall(pool.scores, state.labelled, k=cfg.al.top_k),
        best_score_found=float(np.min(labelled_scores)),
        mean_score_labelled=float(np.mean(labelled_scores)),
        pool_spearman=spearman_rho(pool.scores, prediction.mean),
        val_loss=val_loss,
    )
    logger.info(
        "Round %d (%s, seed %d): %d calls, top-%d recall %.3f, best %.2f kcal/mol, "
        "pool rho %.3f.",
        round_index,
        strategy,
        seed,
        record.oracle_calls,
        cfg.al.top_k,
        record.top_k_recall,
        record.best_score_found,
        record.pool_spearman,
    )
    return record, prediction


def _predict_pool(
    cfg: Config,
    device: torch.device,
    pool: MoleculePool,
    state: CampaignState,
    campaign_root: Path,
    round_index: int,
    config_hash: str,
    seed: int,
) -> UncertainPrediction:
    """Predict mean and sigma over the entire pool for the given round.

    The target scaler comes from the checkpoint rather than being refitted:
    predictions must be inverted with the exact scaler the weights were
    trained against, and refitting on a changed labelled set would shift them.
    """
    round_dir = campaign_root / f"round_{round_index:02d}"

    loader = make_loader(
        pool.subset(range(len(pool))),
        cfg.train,
        shuffle=False,
        seed=seed,
        device_type=device.type,
    )

    if cfg.uncertainty.method == "deep_ensemble":
        from ..models.ensemble import load_ensemble

        members, scaler = load_ensemble(cfg, round_dir, device)
        return predict_with_uncertainty(
            members, loader, device, scaler, method="deep_ensemble"
        )

    from ..models.checkpoint import load_checkpoint

    net = build_model(cfg.model).to(device)
    _, scaler = load_checkpoint(
        round_dir / "best.pt",
        model=net,
        architecture=net.architecture_summary(),
        restore_rng=False,
        map_location=device,
    )
    return predict_with_uncertainty(
        net.to(device),
        loader,
        device,
        scaler,
        method="mc_dropout",
        n_passes=cfg.uncertainty.mc_passes,
        chunk=cfg.uncertainty.mc_chunk,
    )


def records_to_frame(records: list[RoundRecord]) -> pd.DataFrame:
    """Return the per-round records as a dataframe."""
    return pd.DataFrame([record.to_dict() for record in records])


def run_experiment(
    cfg: Config,
    device: torch.device,
    pool: MoleculePool,
    run_root: Path,
    *,
    config_hash: str = "",
    resume: bool = True,
) -> pd.DataFrame:
    """Run every configured strategy across every configured seed.

    Campaigns are independent, so they run sequentially and each checkpoints
    itself. A killed experiment resumes campaign by campaign: finished ones
    replay from their saved state without retraining.

    Returns:
        A long-form dataframe with one row per (strategy, seed, round).
    """
    frames: list[pd.DataFrame] = []
    total = len(cfg.al.strategies) * len(cfg.al.seeds)
    done = 0

    for strategy in cfg.al.strategies:
        for seed in cfg.al.seeds:
            done += 1
            logger.info(
                "=== Campaign %d/%d: strategy=%s seed=%d ===", done, total, strategy, seed
            )
            records = run_campaign(
                cfg,
                device,
                pool,
                strategy=strategy,
                seed=seed,
                run_root=run_root,
                config_hash=config_hash,
                resume=resume,
            )
            frames.append(records_to_frame(records))

    combined = pd.concat(frames, ignore_index=True)
    if "random" not in set(combined["strategy"]):
        # The random baseline is the claim the result has to beat; running
        # without it produces a figure that cannot support any conclusion.
        logger.warning(
            "No random baseline in this experiment. The active-learning curves "
            "cannot be interpreted without it; add 'random' to al.strategies."
        )
    return combined
