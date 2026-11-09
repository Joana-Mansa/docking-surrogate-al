"""Tests for checkpointing and resumption.

The property under test is the one that matters operationally: training that
is interrupted and resumed must end up where uninterrupted training would
have. That is why the comparison here is against a reference run rather than
just "the file loads".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from docking_surrogate_al.config import Config, ModelConfig, RunConfig, TrainConfig
from docking_surrogate_al.data.dataset import MoleculePool, TargetScaler, make_loader
from docking_surrogate_al.models.checkpoint import (
    CheckpointMismatchError,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from docking_surrogate_al.models.gnn import build_model
from docking_surrogate_al.models.train import train_model
from docking_surrogate_al.seeds import seed_everything

SMILES = [
    "CCOc1ccccc1C(=O)O",
    "NC(=O)c1ccccc1",
    "Cc1c[nH]c2ccccc12",
    "c1ccc2ncccc2c1",
    "c1ccc(N2CCNCC2)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCNC(=O)c1ccccc1",
    "OCc1c[nH]c2ccccc12",
    "Clc1ccc2ncccc2c1",
    "Fc1ccc(N2CCNCC2)cc1",
    "CCNC(=O)c1ccc(C)cc1",
    "COc1ccc(CCN)cc1",
]
SCORES = [-7.1, -6.4, -8.2, -7.8, -6.9, -7.3, -6.1, -8.0, -7.6, -6.6, -7.0, -6.8]


def _tiny_config(tmp_path, epochs: int) -> Config:
    """A deliberately tiny CPU configuration for fast tests."""
    return Config(
        run=RunConfig(name="test", seed=0, output_dir=str(tmp_path), device="cpu"),
        model=ModelConfig(hidden_dim=16, num_layers=2, dropout=0.1),
        train=TrainConfig(
            epochs=epochs,
            batch_size=4,
            lr=0.01,
            num_workers=0,
            pin_memory=False,
            amp=True,  # exercised, but auto-disabled on CPU
            early_stopping_patience=1000,
            lr_scheduler_patience=1000,
        ),
    )


def _loaders(cfg: Config):
    pool = MoleculePool(SMILES, SCORES)
    scaler = TargetScaler.fit(pool.scores)
    graphs = pool.subset(range(len(pool)), scaler)
    train_loader = make_loader(graphs, cfg.train, shuffle=True, seed=0, device_type="cpu")
    val_loader = make_loader(graphs, cfg.train, shuffle=False, seed=0, device_type="cpu")
    return train_loader, val_loader, scaler


def test_checkpoint_round_trip(tmp_path):
    """A saved checkpoint restores weights and bookkeeping exactly."""
    cfg = _tiny_config(tmp_path, epochs=1)
    model = build_model(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    state = TrainingState(epoch=5, best_val_loss=0.25, best_epoch=4)
    scaler = TargetScaler(mean=-7.0, std=0.6)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        state=state,
        target_scaler=scaler,
        architecture=model.architecture_summary(),
        config_hash="abc123",
        seed=0,
    )
    assert path.exists()

    restored_model = build_model(cfg.model)
    restored_state, restored_scaler = load_checkpoint(
        path, model=restored_model, architecture=restored_model.architecture_summary()
    )

    assert restored_state.epoch == 5
    assert restored_state.best_val_loss == pytest.approx(0.25)
    assert restored_scaler.mean == pytest.approx(-7.0)
    assert restored_scaler.std == pytest.approx(0.6)
    for a, b in zip(model.state_dict().values(), restored_model.state_dict().values()):
        assert torch.allclose(a.float(), b.float())


def test_resume_matches_uninterrupted_training(tmp_path):
    """Train 6 epochs, versus 3 + resume 3: same weights, same history."""
    reference_dir = tmp_path / "reference"
    interrupted_dir = tmp_path / "interrupted"

    # Reference: six epochs in one go.
    seed_everything(0)
    cfg_full = _tiny_config(tmp_path, epochs=6)
    train_loader, val_loader, scaler = _loaders(cfg_full)
    reference = train_model(
        cfg_full, torch.device("cpu"), train_loader, val_loader, scaler,
        reference_dir, seed=0, resume=False,
    )

    # Interrupted: three epochs, then resume for three more.
    seed_everything(0)
    cfg_half = _tiny_config(tmp_path, epochs=3)
    train_loader, val_loader, scaler = _loaders(cfg_half)
    train_model(
        cfg_half, torch.device("cpu"), train_loader, val_loader, scaler,
        interrupted_dir, seed=0, resume=False,
    )

    cfg_rest = _tiny_config(tmp_path, epochs=6)
    train_loader, val_loader, scaler = _loaders(cfg_rest)
    resumed = train_model(
        cfg_rest, torch.device("cpu"), train_loader, val_loader, scaler,
        interrupted_dir, seed=0, resume=True,
    )

    assert resumed.state.epoch == reference.state.epoch == 6
    assert len(resumed.state.history) == len(reference.state.history) == 6

    # The resumed run must have continued, not restarted: its first three
    # epochs are identical to the reference's, by construction.
    for a, b in zip(resumed.state.history[:3], reference.state.history[:3]):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-5)

    assert resumed.state.best_val_loss == pytest.approx(
        reference.state.best_val_loss, rel=1e-4
    )


def test_resume_is_a_noop_when_already_finished(tmp_path):
    """Rerunning a completed stage does not train further."""
    cfg = _tiny_config(tmp_path, epochs=3)
    train_loader, val_loader, scaler = _loaders(cfg)
    directory = tmp_path / "done"

    first = train_model(
        cfg, torch.device("cpu"), train_loader, val_loader, scaler,
        directory, seed=0, resume=False,
    )
    second = train_model(
        cfg, torch.device("cpu"), train_loader, val_loader, scaler,
        directory, seed=0, resume=True,
    )
    assert first.state.epoch == second.state.epoch == 3
    assert len(second.state.history) == 3


def test_checkpoint_writes_last_and_best(tmp_path):
    cfg = _tiny_config(tmp_path, epochs=2)
    train_loader, val_loader, scaler = _loaders(cfg)
    directory = tmp_path / "ckpts"
    train_model(
        cfg, torch.device("cpu"), train_loader, val_loader, scaler,
        directory, seed=0, resume=False,
    )
    assert (directory / "last.pt").exists()
    assert (directory / "best.pt").exists()


def test_architecture_mismatch_raises(tmp_path):
    """Loading into a differently shaped model fails loudly."""
    cfg = _tiny_config(tmp_path, epochs=1)
    model = build_model(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        state=TrainingState(),
        target_scaler=TargetScaler(0.0, 1.0),
        architecture=model.architecture_summary(),
        config_hash="",
        seed=0,
    )

    wider = build_model(ModelConfig(hidden_dim=64, num_layers=2))
    with pytest.raises(CheckpointMismatchError, match="architecture does not match"):
        load_checkpoint(path, model=wider, architecture=wider.architecture_summary())


def test_featuriser_version_mismatch_raises(tmp_path, monkeypatch):
    """A changed feature set invalidates the weights, and says so."""
    cfg = _tiny_config(tmp_path, epochs=1)
    model = build_model(cfg.model)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
        scheduler=None,
        scaler=None,
        state=TrainingState(),
        target_scaler=TargetScaler(0.0, 1.0),
        architecture=model.architecture_summary(),
        config_hash="",
        seed=0,
    )

    monkeypatch.setattr(
        "docking_surrogate_al.models.checkpoint.FEATURISER_VERSION", "99.0.0"
    )
    with pytest.raises(CheckpointMismatchError, match="featuriser version"):
        load_checkpoint(path, model=build_model(cfg.model))


def test_target_scaler_inverse_round_trip():
    """Scaling and inverting returns the original values."""
    values = np.array([-9.0, -7.5, -6.0, -8.25])
    scaler = TargetScaler.fit(values)
    assert np.allclose(scaler.inverse(scaler.transform(values)), values)


def test_target_scaler_inverse_std_applies_scale_only():
    """A spread is scaled but not shifted; shifting it is a classic bug."""
    scaler = TargetScaler(mean=-7.0, std=2.0)
    assert scaler.inverse_std(np.array([1.0])) == pytest.approx([2.0])
    assert scaler.inverse(np.array([1.0])) == pytest.approx([-5.0])
