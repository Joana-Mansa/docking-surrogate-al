# docking-surrogate-al

A graph neural network that approximates molecular docking scores, and an
uncertainty-guided active-learning loop that reduces the number of expensive
docking calls needed to find top-scoring compounds.

## Scientific framing

**Docking scores are not experimental binding affinities.** A surrogate that
predicts docking well has learned to imitate docking, including its errors.
This project measures surrogate fidelity to the physics oracle and the sample
efficiency of active learning. It makes no claim about real binding affinity
or therapeutic relevance.

Everything downstream follows from that. The target variable is whatever smina
outputs for a given receptor, box and exhaustiveness. If smina is wrong about
a molecule, a perfect surrogate is wrong about it in exactly the same way, and
nothing in this repository would detect that. The questions this code can
answer are "how closely does the surrogate reproduce the oracle" and "how many
oracle calls does the surrogate save", and those are the only questions it
answers.

## Status

The pipeline is implemented and the test suite passes (91 tests). What has and
has not actually been executed:

| Stage | Status |
| --- | --- |
| `prepare-data` | Run on the 200-molecule smoke set. |
| `dock` | **Not run.** No smina binary was available on the development machine (Windows). The docking stage is exercised by CI on Linux, which installs smina from conda-forge and docks the smoke set for real. |
| `train`, `evaluate`, `run-al` | Run end to end on CPU with the smoke config, using placeholder scores standing in for the oracle, so every code path outside `dock` is verified. |
| GPU training | **Not run.** The development machine has a 6 GB RTX 3050 but `smina` was unavailable, so there was no docked pool to train on. The CUDA code paths are implemented and device resolution is tested, but no GPU training run has been performed. |
| Multi-GPU (`uncertainty.ddp`) | **Not implemented.** It raises a clear error. See `src/docking_surrogate_al/models/ensemble.py` for the design and why DDP is the wrong tool for ensemble members. |
| gnina backend | **Not implemented.** A documented stub that refuses to run. |
| AMP parity check | **Not run**, for the same reason as GPU training. The script is `scripts/check_amp_parity.py`; see [Mixed precision](#mixed-precision). |

**No experimental results are reported in this repository.** The numbers the
smoke config produces are not results: 200 molecules, 3 active-learning rounds
and docking exhaustiveness 1 exist to exercise code paths cheaply. They should
never be quoted.

## Pipeline

```
prepare-data  ChEMBL download (or a local CSV) -> RDKit standardisation
     |
   dock       smina against one prepared target, parallel over CPU cores,
     |        every result cached by canonical SMILES
   train      GINE graph network, one model per split, AMP + checkpoints
     |
  evaluate    scaffold and random splits side by side, bootstrap CIs,
     |        calibration of the uncertainty estimates
  run-al      uncertainty / greedy / random acquisition, 10 rounds, 5 seeds
```

### Design choices worth knowing

- **The oracle is an interface.** smina is CPU-only; a CUDA-capable backend
  (gnina) should be addable without touching anything upstream. No module
  outside `oracle/` imports smina. Every cached score records which backend
  and version produced it.
- **The docking cache is an append-only log**, flushed and fsynced per
  molecule, keyed by `(canonical SMILES, settings fingerprint)`. An
  interrupted docking run keeps every molecule that finished, and changing the
  box or exhaustiveness starts a fresh partition rather than silently reusing
  stale scores.
- **Standardisation is the cache key.** It must be deterministic and
  idempotent, or the same compound gets docked twice under two keys. Both
  properties are asserted in the test suite.
- **Dropout sits inside the encoder, not only the head.** Head-only dropout
  would make MC-dropout sample readout noise while the learned representation
  stayed fixed, which produces confident nonsense on exactly the
  out-of-distribution molecules where uncertainty matters.
- **MC-dropout passes are batched.** Each batch is replicated `mc_chunk` times
  into a single forward pass rather than looping passes, with BatchNorm held
  in eval mode so replication cannot leak into batch statistics.
- **Both splits are always reported together.** The gap between scaffold and
  random is the result that justifies the evaluation section.

## Hardware requirements

### GPU

Training and inference run on CUDA. `run.device` is `cuda`, `cpu` or `auto`,
taken from config and never hardcoded. `cuda` is a *requirement*: if CUDA is
unavailable the run fails in the first second with a message naming what torch
sees and what to do about it. Only `auto` is permitted to fall back, and it
logs that it did.

**CUDA version: 12.8.** `pyproject.toml` pins `torch==2.9.1+cu128` from the
PyTorch cu128 index, declared explicitly under `[tool.uv.index]`. torch-geometric
is pure Python and dispatches to whichever torch build is installed, so the
torch pin is what fixes the CUDA version for the whole project. The cu128
wheels need an NVIDIA driver of at least 525 (Linux) or 528 (Windows).

### Defaults and VRAM

The shipped defaults in `configs/default.yaml` target a **~6 GB** card (the
RTX 3050 laptop GPU this was developed on), so a clean clone trains out of the
box on modest hardware. The project was specified against a ~16 GB card;
`configs/gpu_16gb.yaml` is that profile, with a larger batch size, wider
model, more dataloader workers and all MC-dropout passes in one forward call.

`train.batch_size`, `train.num_workers` and `train.pin_memory` are all config
parameters. `pin_memory` is forced off on CPU, where it is pure overhead, and
`num_workers` is forced to 0 on Windows, since the graphs are already in
memory and the spawn start method would re-import RDKit per worker.

The run manifest records GPU name, total VRAM, capability, CUDA version, peak
allocated and reserved VRAM, and samples per second, alongside the CPU timings.

### CPU

smina is CPU-only. `oracle.n_workers` is a **CPU core** count and is unrelated
to any GPU setting; it defaults to `cpu_count() - 1`, leaving one core for the
parent process. Each smina invocation is pinned to `--cpu 1` so that N worker
processes do not each spawn N threads.

**Measured docking cost, and why the smoke test is not five minutes.**
Fitting the two measured configurations (200 invocations at 751s, 8
invocations at 407s, both on 3 workers) separates the cost into roughly
**5.4s of per-invocation receptor setup** and **5.9s per ligand** for
conformer embedding plus search. Batching removes almost all of the first
term: 8 batches measured 407s against a 393s floor at infinite batching. The
second term cannot be removed, so 200 molecules on 3 workers cannot finish
below about 393s. The original five-minute smoke budget was therefore
unachievable with a real oracle at this molecule count. The CI budget is 600s
and the molecule count was kept, rather than shrinking the smoke set until a
number passed.

**Docking throughput and why ligands are batched.** Docking one ligand per
smina process measured at roughly 11 seconds per molecule on a CI runner, and
shrinking the search box by 2.4x changed that by nothing. The cost is
therefore not search: it is the fixed per-invocation work of reading and
atom-typing the receptor and building grid maps, paid once per molecule.
`oracle.ligand_batch_size` (default 25) docks many ligands per invocation
from a multi-model SDF so that setup is amortised across a batch. Set it to 1
to restore one process per ligand.

Results are matched back to molecules by the ligand *name* written into the
SDF, never by output order, because smina omits ligands it fails to dock and
position-based matching would silently shift scores onto the wrong molecules.
`ligand_batch_size` is deliberately *not* part of the docking cache
fingerprint, which assumes batching does not change scores; CI checks that
assumption directly by docking the same molecules batched and unbatched and
comparing (`tests/test_batched_docking.py`).

### No GPU at all

The smoke config is CPU-only and is what CI runs, so the repository is usable
on a machine without a GPU. Install from `requirements-cpu.txt` rather than
`pyproject.toml`: the latter pins the CUDA build deliberately and will not
install without CUDA wheels.

## Mixed precision

`train.amp` defaults to on. Autocast runs the forward pass in fp16 while
`GradScaler` keeps small gradients from flushing to zero; gradients are
unscaled before clipping, so the clip threshold applies to real gradient
magnitudes. On CPU, AMP is **automatically disabled with a log line**, because
CPU autocast uses bfloat16 and is slower than fp32 for a model this size. This
means leaving `amp: true` in a CPU config is harmless rather than a trap.

The claim that AMP does not change results beyond seed variance is *checkable*
rather than asserted: `scripts/check_amp_parity.py` trains the same config with
AMP on and off across several seeds and compares the AMP-vs-FP32 gap in mean
test RMSE to the seed-to-seed standard deviation of the FP32 arm.

```bash
python scripts/check_amp_parity.py --config configs/default.yaml --seeds 3
```

**This check has not been run yet** (see [Status](#status)), so the README
reports no parity figure. Record the output of the script here when a docked
pool and a GPU are both available. The criterion is deliberately loose: it
tests "within seed noise", not bit-identity, which would be false.

## Installation

Requires **Python 3.11** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Joana-Mansa/docking-surrogate-al.git
cd docking-surrogate-al

# GPU (CUDA 12.8)
uv sync

# CPU only
uv venv --python 3.11
uv pip install -r requirements-cpu.txt
```

`smina` is not a Python package and must be installed separately:

```bash
conda install -c conda-forge smina openbabel   # recommended
# or: apt-get install smina
# or: a static build from https://sourceforge.net/projects/smina/
```

Open Babel is optional. If present, the receptor is converted to PDBQT once
(faster, atom typing cached); if absent, smina reads the cleaned PDB directly,
which gives the same scores more slowly, and the fallback is logged.

## Reproduction from a clean clone

Start with the smoke test. It runs the entire pipeline on 200 molecules,
CPU-only, and needs no GPU and no network. Budget about seven minutes on a
3-core machine, nearly all of it real docking:

```bash
export PYTHONPATH=src   # or `uv pip install -e .`
python -m docking_surrogate_al.cli prepare-data --config configs/smoke.yaml
python -m docking_surrogate_al.cli dock         --config configs/smoke.yaml
python -m docking_surrogate_al.cli train        --config configs/smoke.yaml
python -m docking_surrogate_al.cli evaluate     --config configs/smoke.yaml
python -m docking_surrogate_al.cli run-al       --config configs/smoke.yaml
```

Then the full experiment, which needs smina, a GPU and several hours of CPU
time for the docking stage:

```bash
python -m docking_surrogate_al.cli prepare-data --config configs/default.yaml
python -m docking_surrogate_al.cli dock         --config configs/default.yaml
python -m docking_surrogate_al.cli train        --config configs/default.yaml
python -m docking_surrogate_al.cli evaluate     --config configs/default.yaml
python -m docking_surrogate_al.cli run-al       --config configs/default.yaml
```

Scale to 50,000 molecules with `--set data.chembl.n_molecules=50000`. Nothing
else needs to change.

Any stage can be interrupted and rerun; it resumes from the last checkpoint.
Overrides go through the same validation as the config file:

```bash
python -m docking_surrogate_al.cli train --config configs/default.yaml \
  --set train.batch_size=256 --set uncertainty.mc_chunk=30
```

### Outputs

- `runs/<name>/metrics.csv` — long form, one row per split and metric, with
  bootstrap CIs.
- `runs/<name>/active_learning.csv` — one row per (strategy, seed, round).
- `runs/<name>/predictions_<split>.csv` — per-molecule predictions and sigma.
- `runs/<name>/manifest.json` — config hash, git commit and dirty flag,
  package versions, hardware, seed, and wall-clock time, throughput and peak
  VRAM per stage.
- `figures/` — all figures at 300 dpi.
- `cache/docking_cache.jsonl` — every docking result, with its backend.

## The target

Trypsin with benzamidine bound (PDB **3PTB**), box centred on the
co-crystallised benzamidine at `(-1.76, 14.46, 16.92)`.

It was chosen for being small, fast to dock and a long-standing docking
validation system, **not** for therapeutic interest. The ChEMBL molecules
screened against it are drug-like compounds with no particular relationship to
trypsin. That is consistent with the framing above: the experiment measures
whether a surrogate can imitate an oracle, and for that the target only has to
be a well-defined, reproducible scoring function. Target, box and
exhaustiveness are all config parameters.

## Testing

```bash
PYTHONPATH=src pytest tests -q
```

91 tests covering data standardisation (including idempotence, since the
canonical SMILES is the cache key), scaffold splitting (asserting no scaffold
overlap between train and test, and that the guard fires on a deliberately
leaky split), every metric against hand-computed values, calibration against
synthetic calibrated and miscalibrated predictions, acquisition strategies,
the docking cache under simulated interruption, config validation, and
checkpoint/resume — the last comparing a 3-epoch-plus-resume run against an
uninterrupted 6-epoch run, not merely checking that a file loads.

## Limitations

- **Single target.** One protein, one box, one exhaustiveness setting.
  Nothing here shows the approach transfers to another target, and docking
  surrogates are known to be target-specific.
- **Docking as ground truth.** The oracle is smina, not an experiment. The
  surrogate is trained to reproduce the oracle's errors as faithfully as its
  successes. Reported fidelity is fidelity *to smina*.
- **No experimental validation.** Nothing in this repository has been checked
  against measured binding affinity, and no molecule has been synthesised or
  assayed.
- **Scaffold split is still optimistic.** It removes the most obvious leakage,
  but train and test molecules still come from the same library, prepared the
  same way, docked against the same receptor with the same box. True
  prospective use means new chemistry from a different source, which is harder
  than anything measured here.
- **Small chemical space.** 5,000 molecules by default (50,000 at the
  configured ceiling) is a narrow slice of drug-like space. Enrichment at 1%
  on a few hundred test molecules is coarse: the top 1% may be two molecules,
  so that metric takes few distinct values and has a wide, lumpy confidence
  interval. `n_test` is reported alongside every metric so this is visible.
- **Retrospective active learning.** The pool is docked once up front, and
  "calling the oracle" reveals a precomputed label. This makes 150 campaigns
  affordable and removes docking noise between strategies, but it cannot
  discover molecules outside the pool. The curves measure sample efficiency
  *within a fixed library*.
- **Uncertainty is approximate.** MC-dropout approximates a posterior over
  weights and captures epistemic uncertainty only; it does not model the
  oracle's own variability. This is why calibration is measured and reported
  rather than assumed.
- **Single starting conformer per ligand.** smina searches ligand torsions
  internally, but a single ETKDG conformer is still an input choice that
  affects scores.
- **Conservative receptor preparation.** Waters and heteroatoms are stripped;
  missing loops are not modelled, protonation states are not assigned per
  residue, and side chains are rigid.
- **Seed variance bands are single-machine.** They bound rerun-to-rerun
  variation on one GPU with one library stack, not variation across machines,
  drivers or library versions.

## Licence

MIT. See [LICENSE](LICENSE).
