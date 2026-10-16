# Running and resuming the Python model

The Python package is the baseline locomotor adaptation implementation. Its seeded examples and checkpoints are **not participant fits or checkpoints from the paper figures**. Use the publication commands and reference-data manifests for those figures. The separate manuscript MATLAB engine preserves the supplied simulation settings.

## Run an example

Create `loco-ada` using the [installation instructions](../README.md#get-started). The Conda environment includes the editable package, pinned scientific dependencies and tests. Activate it before running the commands below:

```sh
conda activate loco-ada
loco-simulate demo
loco-simulate run --seed 42 --iterations 1000 --output-dir outputs/example
```

In a shell without Conda activation, use `conda run --no-capture-output -n loco-ada loco-simulate` with the same arguments. The scientific dependencies are pip-installed inside Conda from `requirements-lock.txt`; keep them synchronized through `conda env update -n loco-ada -f environment.yml`.

The demo runs 40 iterations with seed 42 and saves to `outputs/demo/`. `run` accepts the full scientific options and saves to `outputs/run/` unless you choose another directory. Both commands save checkpoints every 25 iterations and at the end. Figures are saved without opening windows; add `--show-plots` to open them. `python -m locomotor_learning_model` accepts the same commands as `loco-simulate`.

An output directory receives `run.npz`, `metadata.json`, `checkpoint.npz`, `checkpoint.metadata.json` and `split_belt_summary.png` (or `tied_belt_summary.png`). The result archive contains controller histories, body trajectories, per-step energy/work/timing, learner diagnostics, parameters, and summary arrays. Metadata records the actual seed, numerical-library versions, scientific-configuration hash, and completed/target iteration counts. An omitted seed is generated and recorded.

`--split-or-tied tied` selects tied walking. The baseline defaults remain a 1.0 m/s speed difference with a 15-second transition for split walking, or the four-speed-changes protocol with a 3-second transition for tied walking. Protocol overrides are validated. The default learner has two physical steps per learning iteration. Short examples may finish before the protocol reaches split walking; use the saved times and treadmill schedule to identify the portion simulated.

## Pause and resume

```sh
loco-simulate run --seed 42 --iterations 100 --stop-after 40 --output-dir outputs/paused-example
loco-simulate resume outputs/paused-example/checkpoint.npz
```

`--iterations` is the **total target**, not an additional count. Omitting it on resume uses the saved target. It may be increased; it cannot be less than the completed count. `--stop-after` is also an absolute completed-iteration count. The new commands supply a checkpoint path automatically. A checkpoint is saved at each requested interval and at the last completed learning iteration. Interruption between saves loses only work since the last save.

`resume` saves a new checkpoint in `outputs/resumed/` by default and preserves the input checkpoint. Choose another `--output-dir` for subsequent branches of a run; use `--checkpoint PATH` when deliberately updating a specific checkpoint. Set `--checkpoint-every 10` to save more frequently. Existing flag-only commands remain supported.

Checkpoints capture the body state, simulation time and treadmill context, current controller and gradient, learned memory, rolling estimator matrices and input histories, all diagnostic histories, controller gains, parameters, and the complete MT19937 state including the cached Gaussian value. State is saved **after a complete learning iteration**, not inside an ODE solve. Post-processing subsequently replays the learned controller sequence from the initial body state; this replay is redone on resume. A result archive can be read and plotted without replaying or training.

The archive uses numeric arrays and JSON with `allow_pickle=False`; it does not execute serialized Python objects. Writes use a temporary file and atomic replacement. Schema, model version, scientific configuration, history lengths/shapes, finite state, and RNG state are checked before resuming. A changed protocol or seed is rejected. Exact resume requires the recorded NumPy and SciPy versions; results can still be loaded with other versions for inspection. Bitwise identity is tested on the same numerical runtime, not promised across different hardware/BLAS implementations.

The pause/resume commands above create their own checkpoint and need no data download. Optional saved examples are distributed in the DATA release, outside the GitHub code repository. After following the [data setup instructions](../README.md#reproduce-the-paper-figures), `data/locomotor_learning_model-DATA/checkpoints/example_seed42.npz` provides a baseline split-protocol example paused after 10 iterations with a target of 20. Its adjacent metadata identifies its role. It demonstrates continuation; it is not a trained neural network or a paper result:

```sh
loco-simulate resume data/locomotor_learning_model-DATA/checkpoints/example_seed42.npz --output-dir outputs/checkpoint-example
```

If DATA is beside the repository, use `../locomotor_learning_model-DATA/checkpoints/example_seed42.npz` as the checkpoint path instead.

The same DATA `checkpoints/` folder contains `manuscript_seed42_trial.mat` and its metadata: a completed seeded manuscript-engine trial that terminated at a detected model fall on iteration 104. This is a reusable numerical result, not a resumable Python learner or a published fitted model. The [manuscript engine guide](../matlab/manuscript/README.md) explains its API.

The Conda environment file installs the NumPy/SciPy versions recorded in the Python example. If your active versions differ, update and activate `loco-ada` before resuming. With an intentionally different numerical runtime, create a fresh local checkpoint with the pause command above instead. Do not edit the recorded runtime version to bypass compatibility checks.

## Python API and replotting

```python
from locomotor_learning_model import (
    run_simulation, load_simulation_results, plot_simulation_results,
)

result = run_simulation(seed=42, num_iterations=100, make_plots=False,
                        checkpoint_path="outputs/run/checkpoint.npz",
                        checkpoint_every=25, output_dir="outputs/run")
saved = load_simulation_results("outputs/run/run.npz")
print(saved.summary["average_energy_rate"])
plot_simulation_results(saved, output_dir="outputs/replotted", show_plots=False)
```

The high-level API owns a private random generator and does not reseed NumPy's global generator. Low-level functions retain the historical global-RNG behavior for existing callers unless a runtime generator is provided internally. Do not interchange low-level diagnostic dictionaries with versioned workflow checkpoints.

## Scientific scope and validation

The baseline model equations and learner/noise/memory defaults are preserved. The manuscript engine uses different settings: it divides belt speeds by 2, sets a 25-second transition, disables the baseline memory-learning rates, and uses another learning-rate scale. The manuscript describes Froude-like speed scaling and a 20-second transition. These source/Methods discrepancies remain unresolved. Original fitted MAT files, regenerated trial-averaged grids, and new seeded examples have distinct provenance.

Baseline Python/MATLAB deterministic parity can be checked with:

```matlab
addpath('matlab/validation');
if ~isfolder('outputs'), mkdir('outputs'); end
export_reference_run('outputs/locad_reference.mat', 40);
```

```sh
python scripts/validate_matlab_python_parity.py outputs/locad_reference.mat
python -m pytest tests/test_checkpoints.py tests/test_detailed_function_comparison.py tests/test_matlab_python_equivalence.py
```

The MATLAB comparison disables exploratory and energy-measurement noise; identical seeds do not establish cross-language stochastic identity. Forty iterations exercise the rolling estimator after its initial 30-iteration window. The test suite also compares uninterrupted and resumed stochastic trajectories across split/tied protocol changes, repeated seeds, safe result reload/replotting, input/configuration rejection, and atomic-write failures.
