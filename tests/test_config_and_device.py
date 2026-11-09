"""Tests for config loading, hashing and device resolution."""

from __future__ import annotations

import pytest
import torch

from docking_surrogate_al.config import (
    Config,
    ConfigError,
    REPO_ROOT,
    config_hash,
    load_config,
    resolve_path,
)
from docking_surrogate_al.device import DeviceUnavailableError, resolve_device


def _write(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_shipped_configs():
    """Every committed config must parse against the current schema."""
    for name in ("default.yaml", "smoke.yaml", "gpu_16gb.yaml"):
        cfg = load_config(REPO_ROOT / "configs" / name)
        assert isinstance(cfg, Config)


def test_smoke_config_is_cpu_only_and_offline():
    """The two properties CI depends on, asserted rather than assumed."""
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    assert cfg.run.device == "cpu"
    assert cfg.data.source == "csv"
    assert cfg.data.csv_path is not None
    assert resolve_path(cfg.data.csv_path).exists()


def test_default_config_includes_the_random_baseline():
    """The baseline is mandatory; a config without it is a broken experiment."""
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert "random" in cfg.al.strategies


def test_unknown_key_is_rejected(tmp_path):
    """A typo must fail at load time, not revert to a default silently."""
    path = _write(tmp_path, "train:\n  batch_sze: 32\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_wrong_type_is_rejected(tmp_path):
    path = _write(tmp_path, "train:\n  batch_size: sixty-four\n")
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_overrides_are_applied_and_validated(tmp_path):
    path = _write(tmp_path, "train:\n  epochs: 100\n")
    cfg = load_config(path, {"train.epochs": 2, "run.name": "ovr"})
    assert cfg.train.epochs == 2
    assert cfg.run.name == "ovr"

    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(path, {"train.epochs": "many"})


def test_tuple_fields_are_coerced(tmp_path):
    path = _write(tmp_path, "target:\n  center: [1.0, 2.0, 3.0]\nal:\n  seeds: [0, 1]\n")
    cfg = load_config(path)
    assert cfg.target.center == (1.0, 2.0, 3.0)
    assert cfg.al.seeds == (0, 1)


def test_config_hash_is_stable_and_sensitive():
    """Same resolved config, same hash; any change, different hash."""
    a = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    b = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    c = load_config(REPO_ROOT / "configs" / "smoke.yaml", {"train.epochs": 99})

    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash(a)) == 64


def test_resolve_path_is_repo_relative():
    assert resolve_path("configs/default.yaml") == REPO_ROOT / "configs" / "default.yaml"


def test_device_cpu_is_always_available():
    assert resolve_device("cpu") == torch.device("cpu")


def test_device_rejects_unknown_values():
    with pytest.raises(DeviceUnavailableError, match="must be one of"):
        resolve_device("tpu")


def test_device_auto_never_raises():
    """auto is the only mode allowed to fall back."""
    assert resolve_device("auto").type in {"cpu", "cuda"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_device_cuda_fails_loudly_without_a_gpu():
    """The failure message must say what to do, not just that it failed."""
    with pytest.raises(DeviceUnavailableError) as excinfo:
        resolve_device("cuda")
    message = str(excinfo.value)
    assert "no CUDA device" in message
    assert "cu128" in message or "nvidia-smi" in message
