# docking-surrogate-al

[![CI](https://github.com/Joana-Mansa/docking-surrogate-al/actions/workflows/ci.yml/badge.svg)](https://github.com/Joana-Mansa/docking-surrogate-al/actions/workflows/ci.yml)

A graph neural network that predicts molecular docking scores, and an
uncertainty-guided active-learning loop that reaches the top-scoring compounds
in a library using fewer docking calls than random selection.

Docking is expensive: tens of CPU-seconds per molecule, so a million-compound
screen is CPU-months. This measures two things. How faithfully can a GNN
reproduce the docking oracle on unseen scaffolds, and how many oracle calls
does uncertainty-guided selection save against a random baseline.

> **Docking scores are not experimental binding affinities.** A surrogate that
> predicts docking well has learned to imitate docking, including its errors.
> This project makes no claim about real binding affinity or therapeutic
> relevance.

## Quickstart

```bash
git clone https://github.com/Joana-Mansa/docking-surrogate-al.git
cd docking-surrogate-al
uv sync                                    # GPU, CUDA 12.8
conda install -c conda-forge smina openbabel

export PYTHONPATH=src
for stage in prepare-data dock train evaluate run-al; do
  python -m docking_surrogate_al.cli $stage --config configs/smoke.yaml
done
```

The smoke config runs the whole pipeline on 200 molecules, CPU-only, no
network, in about seven minutes. It is what CI runs on every push. For a CPU
install use `requirements-cpu.txt`; `pyproject.toml` pins a CUDA build.

## Pipeline

```
prepare-data  ChEMBL download (or local CSV) -> RDKit standardisation
     |
   dock       smina against one target, parallel over CPU cores,
     |        every result cached by canonical SMILES
   train      GINE graph network, one model per split, AMP + checkpoints
     |
  evaluate    scaffold and random splits side by side, bootstrap CIs,
     |        calibration of the uncertainty estimates
  run-al      uncertainty / greedy / random acquisition, 10 rounds, 5 seeds
```

Five CLI entry points, one YAML config, no hardcoded paths or hyperparameters.
Any stage can be interrupted and rerun; it resumes from the last checkpoint.

```bash
python -m docking_surrogate_al.cli train --config configs/default.yaml \
  --set train.batch_size=256 --set uncertainty.mc_chunk=30
```

## Evaluation

Every metric is reported on **both** a Bemis-Murcko scaffold split and a random
split. The random split flatters the model; reporting only that number would be
the easy lie. A test asserts no scaffold appears on both sides of the boundary.

| | |
|---|---|
| Fidelity | RMSE, Spearman rho |
| Ranking | enrichment factor at 1% and 5%, precision@{10,50,100} |
| Uncertainty | reliability curve, ECE, error-vs-sigma correlation |
| Confidence | non-parametric paired bootstrap, 1000 resamples, 95% CI |

The bootstrap resamples molecules rather than residuals, which is what keeps
the rank metrics valid. Active learning runs 3 strategies x 5 seeds x 10
rounds, with the random baseline plotted in the same weight as the others.

## Design notes

- **The oracle is an interface.** smina is CPU-only; a CUDA backend (gnina)
  can be added without touching anything upstream. Nothing outside `oracle/`
  imports smina, and every cached score records which backend produced it.
- **The docking cache is an append-only log**, fsynced per record and keyed by
  `(canonical SMILES, settings fingerprint)`. Interrupted runs keep everything
  finished; changing the box or exhaustiveness starts a fresh partition instead
  of silently reusing stale scores.
- **Ligands are batched per smina call.** Receptor setup dominates per-molecule
  cost, so `oracle.ligand_batch_size` amortises it. Results are matched back by
  ligand name, never output order, because smina omits ligands it fails to
  dock. CI verifies batched and unbatched scores agree.
- **Dropout sits inside the encoder, not only the head**, so MC-dropout samples
  representation uncertainty rather than readout noise. Passes are batched into
  single forward calls with BatchNorm held in eval.
- **Positive docking scores are steric clashes, not weak binding.** Left in,
  a handful dominate the target scaler. `data.max_docking_score` excludes them
  at pool construction, the cache keeps what the oracle actually returned, and
  every run reports how many were dropped.
- **Standardisation is the cache key**, so it must be deterministic and
  idempotent. Both are asserted in tests.

## Hardware

`run.device` is `cuda`, `cpu` or `auto`, taken from config and never
hardcoded. `cuda` is a requirement: if CUDA is missing the run fails in the
first second with a message naming what torch sees. Only `auto` falls back, and
it logs that it did.

**CUDA 12.8**, pinned via `torch==2.9.1+cu128` from the PyTorch cu128 index.
Driver 525+ (Linux) or 528+ (Windows).

`configs/default.yaml` targets a ~6 GB card so a clean clone trains out of the
box; `configs/gpu_16gb.yaml` is the larger profile. Batch size, dataloader
workers and pin_memory are all config parameters.

Mixed precision defaults on, with gradients unscaled before clipping, and is
auto-disabled on CPU where bfloat16 autocast is slower than fp32.
`scripts/check_amp_parity.py` trains with AMP on and off across seeds and
compares the gap to seed-to-seed variance, so the "AMP changes nothing" claim
is checkable rather than asserted.

smina is CPU-only. `oracle.n_workers` is a CPU core count, defaulting to
`cpu_count() - 1`, and each invocation is pinned to `--cpu 1`.

## Outputs

Every run writes a manifest recording the config hash, git commit and dirty
flag, package versions, GPU name, VRAM, CUDA version, peak memory, throughput
and per-stage wall clock, so any result can be traced to what produced it.

```
runs/<name>/metrics.csv              one row per split and metric, with CIs
runs/<name>/active_learning.csv      one row per (strategy, seed, round)
runs/<name>/predictions_<split>.csv  per-molecule prediction and sigma
runs/<name>/manifest.json            provenance and hardware
figures/                             all figures at 300 dpi
cache/docking_cache.jsonl            every docking result, with its backend
```

## The target

Trypsin with benzamidine bound (PDB **3PTB**), box centred on the
co-crystallised ligand. Chosen for being small, fast to dock and a standard
docking validation system, not for therapeutic interest. Target, box and
exhaustiveness are config parameters.

## Testing

```bash
PYTHONPATH=src pytest tests -q        # 106 tests
```

Covers standardisation idempotence (it is the cache key), scaffold-split
leakage in both directions, every metric against hand-computed values,
calibration against synthetic calibrated and miscalibrated predictions,
acquisition strategies, the cache under simulated interruption, and
checkpoint/resume compared against an uninterrupted run rather than merely
checking a file loads.

CI runs the unit tests and, separately, the entire pipeline against a real
smina binary installed from conda-forge. It is an integration test, not a mock.

## Not implemented

Both of these raise a clear error rather than failing quietly:

- **Multi-GPU ensemble training** (`uncertainty.ddp`). DDP all-reduces
  gradients across ranks, which would converge ensemble members into one model
  and collapse the predictive spread the ensemble exists to measure. The right
  design is member-parallel, which could not be tested on a single-GPU machine,
  so it is specified in `models/ensemble.py` and left unshipped.
- **The gnina backend.** A documented stub showing where a CUDA-capable oracle
  would plug in.

## Limitations

- **One target, one box, one exhaustiveness.** Nothing here shows the approach
  transfers; docking surrogates are known to be target-specific.
- **The oracle is smina, not an experiment.** Reported fidelity is fidelity to
  smina, including its errors. No molecule has been synthesised or assayed.
- **Scaffold split is still optimistic.** Train and test molecules come from
  the same library, prepared the same way, docked against the same receptor.
  Prospective use means new chemistry from a different source.
- **Active learning is retrospective.** The pool is docked up front and an
  "oracle call" reveals a precomputed label. This makes 150 campaigns
  affordable but cannot discover molecules outside the pool, so the curves
  measure sample efficiency within a fixed library.
- **Clash scores are excluded, not modelled.** Metrics therefore describe
  molecules that dock at all, while a real campaign still pays for the ones
  that do not.
- **Uncertainty is epistemic only.** MC-dropout does not model the oracle's own
  variability, which is why calibration is measured rather than assumed.
- **Small chemical space.** A few thousand molecules is a narrow slice of
  drug-like space, and enrichment at 1% on a few hundred test molecules is
  coarse. `n_test` is reported alongside every metric so this stays visible.

## Licence

MIT. See [LICENSE](LICENSE).
