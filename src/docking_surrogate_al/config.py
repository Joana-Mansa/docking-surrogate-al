"""Typed configuration objects loaded from YAML.

Design notes
------------
Every tunable in this project lives in a YAML file; nothing is hardcoded in
the modules that consume it. The configuration is parsed into frozen
dataclasses rather than kept as a raw dict for three reasons:

1. Typos fail loudly. ``_build`` rejects unknown keys, so a misspelled
   ``batch_sze`` is an error at load time rather than a silently ignored
   setting that makes a run quietly non-reproducible.
2. Static typing. Consumers get real attribute access and mypy coverage.
3. Hashing. :func:`config_hash` hashes the *resolved* configuration, so the
   run manifest records exactly what ran, including defaults that the YAML
   file never mentioned.

Paths in the YAML are resolved relative to the repository root (the parent of
``src/``) unless absolute, so a config file behaves the same no matter which
directory the CLI is invoked from.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Section schemas
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Top-level run identity and output location."""

    name: str = "default"
    seed: int = 0
    output_dir: str = "runs"
    # "cuda", "cpu" or "auto". Never inferred silently anywhere else in the
    # codebase; see docking_surrogate_al.device.resolve_device.
    device: str = "auto"


@dataclass(frozen=True)
class ChemblConfig:
    """Parameters for the ChEMBL download."""

    # Default 5000; raise to 50000 for the full-scale experiment.
    n_molecules: int = 5000
    page_size: int = 1000
    min_mw: float = 200.0
    max_mw: float = 500.0
    max_alogp: float = 5.0
    max_ro5_violations: int = 0
    timeout_s: int = 120
    retries: int = 4


@dataclass(frozen=True)
class DataConfig:
    """Where molecules come from and how they are standardised."""

    # "chembl" downloads from the ChEMBL REST API. "csv" reads a local file,
    # which is how the smoke config stays hermetic and offline-reproducible.
    source: str = "chembl"
    csv_path: str | None = None
    smiles_column: str = "smiles"
    id_column: str | None = "chembl_id"
    raw_path: str = "data/raw/molecules_raw.csv"
    processed_path: str = "data/processed/molecules.csv"
    chembl: ChemblConfig = field(default_factory=ChemblConfig)
    # Standardisation switches, all applied in data.standardise.
    strip_salts: bool = True
    neutralise: bool = True
    max_heavy_atoms: int = 60
    min_heavy_atoms: int = 8


@dataclass(frozen=True)
class TargetConfig:
    """The protein target and docking box.

    The box is given explicitly rather than derived, so a run is reproducible
    even if the receptor file is re-prepared with a different tool version.
    """

    name: str = "trypsin_3ptb"
    receptor_pdb: str = "data/targets/trypsin_3ptb/receptor_raw.pdb"
    receptor_pdbqt: str = "data/targets/trypsin_3ptb/receptor.pdbqt"
    center: tuple[float, float, float] = (-1.76, 14.46, 16.92)
    size: tuple[float, float, float] = (20.0, 20.0, 20.0)


@dataclass(frozen=True)
class OracleConfig:
    """Docking oracle settings.

    smina is CPU-only, so ``n_workers`` is a *CPU core* count. It has nothing
    to do with the GPU settings in :class:`TrainConfig`.
    """

    backend: str = "smina"
    smina_binary: str = "smina"
    gnina_binary: str = "gnina"
    exhaustiveness: int = 8
    num_modes: int = 9
    seed: int = 0
    # None means "all available cores minus one".
    n_workers: int | None = None
    # Ligands docked per smina invocation. smina re-reads and re-types the
    # receptor on every call, which measured at roughly 11s per molecule when
    # docking one at a time; batching amortises that setup across a batch.
    # 1 restores the old one-process-per-ligand behaviour.
    ligand_batch_size: int = 25
    timeout_s: int = 600
    cache_path: str = "cache/docking_cache.jsonl"
    # 3D conformer generation for ligands, done once per molecule with RDKit.
    embed_max_attempts: int = 10
    ff_max_iters: int = 500


@dataclass(frozen=True)
class ModelConfig:
    """GNN architecture."""

    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.15
    # Dropout applied inside message passing as well as the head; MC-dropout
    # needs stochasticity in the encoder, not only the readout.
    pool: str = "mean_max"
    residual: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and GPU execution.

    Defaults target a single ~6 GB laptop GPU (RTX 3050) so that a clean clone
    trains out of the box on modest hardware. ``configs/gpu_16gb.yaml``
    overrides these for the ~16 GB card the project was specified against.
    """

    epochs: int = 150
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 4
    pin_memory: bool = True
    # Mixed precision is on by default and is a no-op on CPU, where autocast
    # for bf16/fp16 would otherwise slow things down; see models.train.
    amp: bool = True
    grad_clip: float = 5.0
    early_stopping_patience: int = 25
    lr_scheduler_patience: int = 10
    checkpoint_every_epoch: bool = True


@dataclass(frozen=True)
class UncertaintyConfig:
    """Predictive uncertainty method."""

    # "mc_dropout" or "deep_ensemble".
    method: str = "mc_dropout"
    mc_passes: int = 30
    # MC-dropout passes are batched on GPU; this caps how many passes are
    # replicated into one forward batch at a time (VRAM bound).
    mc_chunk: int = 10
    ensemble_size: int = 5
    # Deep-ensemble members train sequentially on one GPU by default.
    ddp: bool = False
    ddp_world_size: int = 1


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol."""

    splits: tuple[str, ...] = ("scaffold", "random")
    test_frac: float = 0.2
    val_frac: float = 0.1
    bootstrap_samples: int = 1000
    bootstrap_ci: float = 0.95
    ef_fractions: tuple[float, ...] = (0.01, 0.05)
    precision_k: tuple[int, ...] = (10, 50, 100)
    calibration_bins: int = 10


@dataclass(frozen=True)
class ALConfig:
    """Active-learning loop."""

    initial_size: int = 500
    batch_size: int = 250
    rounds: int = 10
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    # The random baseline is mandatory: it is the claim the result must beat.
    strategies: tuple[str, ...] = ("uncertainty", "greedy", "random")
    top_k: int = 100
    # Epochs per retraining round; usually shorter than a full training run.
    epochs_per_round: int = 60


@dataclass(frozen=True)
class Config:
    """Full resolved configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    al: ALConfig = field(default_factory=ALConfig)

    # -- derived paths ----------------------------------------------------
    @property
    def run_dir(self) -> Path:
        """Directory holding every artefact produced by this run."""
        return resolve_path(self.run.output_dir) / self.run.name

    def path(self, relative: str) -> Path:
        """Resolve ``relative`` against the repository root."""
        return resolve_path(relative)

    def to_dict(self) -> dict[str, Any]:
        """Return the fully resolved configuration as plain data."""
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path against the repository root unless absolute."""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


class ConfigError(ValueError):
    """Raised when a configuration file does not match the schema."""


def _build(cls: type, data: Any, where: str) -> Any:
    """Recursively construct dataclass ``cls`` from ``data``."""
    if not dataclasses.is_dataclass(cls):
        return _coerce(cls, data, where)
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. Valid keys: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _build(hints[f.name], data[f.name], f"{where}.{f.name}")
    return cls(**kwargs)


def _coerce(annotation: Any, value: Any, where: str) -> Any:
    """Coerce a scalar/sequence YAML value to its annotated type."""
    origin = get_origin(annotation)

    # Optional[X] / X | None
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        return _coerce(args[0], value, where)

    if origin is tuple:
        args = get_args(annotation)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, where) for v in value)
        if len(args) != len(value):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(a, v, where) for a, v in zip(args, value, strict=True))

    if origin is list:
        (arg,) = get_args(annotation)
        return [_coerce(arg, v, where) for v in value]

    if dataclasses.is_dataclass(annotation):
        return _build(annotation, value, where)

    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected a boolean, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file into a :class:`Config`.

    ``overrides`` is a flat mapping of dotted keys to values, e.g.
    ``{"train.epochs": 2}``, applied after the file is read. This is how the
    CLI exposes ``--set`` without letting ad-hoc values escape the schema:
    overrides go through the same validation as the file.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    for dotted, value in (overrides or {}).items():
        cursor = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {dotted!r}: {part} is not a section")
            cursor = nxt
        cursor[parts[-1]] = value

    return _build(Config, raw, "config")


def config_hash(cfg: Config) -> str:
    """Stable SHA-256 of the fully resolved configuration.

    Recorded in every run manifest. Two runs with the same hash saw the same
    resolved settings, including defaults absent from the YAML file.
    """
    payload = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
