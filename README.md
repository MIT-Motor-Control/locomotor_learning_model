# locomotor_learning_model

Code accompanying the manuscript "Fall risk-aware adaptation explains suboptimal locomotor performance."

This repository now contains two synchronized implementations of the model:

- `matlab/`: the original MATLAB workflow used during model development.
- `src/locomotor_learning_model/`: a packaged Python implementation with matching default behavior.

The associated bioRxiv preprint is available here:
[Fall risk-aware adaptation explains suboptimal locomotor performance](https://www.biorxiv.org/content/10.64898/2026.03.02.709033v1.abstract)

## Repository layout

```text
locomotor_learning_model/
├── matlab/
│   ├── run_simulation.m
│   ├── src/
│   └── validation/
├── notebooks/
├── scripts/
├── src/locomotor_learning_model/
├── tests/
├── pyproject.toml
└── README.md
```

Key locations:

- `matlab/run_simulation.m`: MATLAB entry point.
- `matlab/validation/export_reference_run.m`: exports a deterministic MATLAB reference run for parity checks.
- `scripts/run_python_simulation.py`: Python entry point without installation.
- `scripts/validate_matlab_python_parity.py`: compares Python against a MATLAB reference export.
- `src/locomotor_learning_model/parameter_loading/load_protocol_parameters.py`: default protocol configuration.
- `src/locomotor_learning_model/workflow.py`: high-level Python workflow.

## Requirements

### Python

- Python 3.10 or newer
- `numpy`
- `scipy`
- `matplotlib`

Install the Python package in editable mode:

```bash
python3 -m pip install -e .
```

### MATLAB

- MATLAB with standard ODE functionality (`ode45`)
- The default simulation does not require a custom toolbox configuration

The MATLAB code was refactored into `matlab/` so it can be run directly without browsing through the original flat code drop.

## Quick start

### Run the Python implementation

Recommended:

```bash
python3 -m pip install -e .
locomotor-learning-model --no-plots
```

Or, without installation:

```bash
python3 scripts/run_python_simulation.py --no-plots
```

Useful options:

- `--iterations 100`: override the default number of learning iterations.
- `--output-dir outputs/python`: save figures as PNG files.
- `--split-or-tied tied --speed-protocol 'single speed change'`: switch to a tied-belt protocol.
- `--seed 42`: make the run reproducible.

### Run the MATLAB implementation

From within MATLAB:

```matlab
cd matlab
run_simulation
```

This wrapper adds `matlab/src` to the path and launches the original manuscript pipeline.

## MATLAB/Python parity

The Python code was checked against the local MATLAB implementation using a deterministic, no-noise reference run. The core controller trajectory matches to machine precision for that reference case.

To reproduce that check:

1. Export a deterministic MATLAB reference:

```matlab
cd matlab/validation
export_reference_run('/tmp/locomotor_learning_model_reference.mat', 10)
```

2. Compare it against Python:

```bash
python3 scripts/validate_matlab_python_parity.py /tmp/locomotor_learning_model_reference.mat
```

Additional Python-side validation scripts:

```bash
python3 tests/test_matlab_python_equivalence.py
python3 tests/test_detailed_function_comparison.py
```

## Configuring the simulations

The default manuscript protocol is the classic split-belt experiment.

Main parameter files:

- Python protocol and duration: `src/locomotor_learning_model/parameter_loading/load_protocol_parameters.py`
- Python learner settings: `src/locomotor_learning_model/parameter_loading/load_learner_parameters.py`
- MATLAB protocol and duration: `matlab/src/loadProtocolParameters.m`
- MATLAB learner settings: `matlab/src/loadLearnerParameters.m`

The Python implementation now includes the tied-belt protocol utilities that existed in MATLAB but were not previously exposed in the refactored Python tree.

## Notes

- `notebooks/rootSimulateLearningWhileWalking.ipynb` is kept as an exploratory notebook version of the Python workflow.
- The MATLAB source filenames remain close to the original manuscript code for traceability.
- The Python package adds a cleaner public interface on top of the original function-level translation.

## Citation

If you use this repository, please cite:

```bibtex
@article{kang2026fall,
  title={Fall risk-aware adaptation explains suboptimal locomotor performance},
  author={Kang, Inseung and Mitra, Kanishka and Seethapathi, Nidhi},
  journal={bioRxiv},
  pages={2026--03},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
