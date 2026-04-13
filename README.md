# locomotor_learning_model

Code accompanying the manuscript "Fall risk-aware adaptation explains suboptimal locomotor performance."

This repository contains two matched implementations of the same model:

- `matlab/`: MATLAB version
- `src/locomotor_learning_model/`: Python version

## Repository layout

```text
locomotor_learning_model/
├── environment.yml
├── matlab/
│   ├── run_simulation.m
│   ├── src/
│   └── validation/
├── scripts/
├── src/locomotor_learning_model/
├── tests/
└── README.md
```

Most important files:

- `environment.yml`: Conda environment for the Python code
- `scripts/run_python_simulation.py`: main Python entry point
- `scripts/validate_matlab_python_parity.py`: Python/MATLAB parity check
- `matlab/run_simulation.m`: main MATLAB entry point
- `matlab/validation/export_reference_run.m`: deterministic MATLAB reference export

## Python setup

### If you do not have Conda yet

Install Miniconda or Miniforge by following the official Conda installation guide:
[Conda installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

Typical flow:

1. Download the installer for your operating system.
2. Run the installer and accept the default options.
3. Close and reopen your terminal.
4. Check that Conda is available:

```bash
conda --version
```

### Create the Conda environment

This repository includes a ready-to-use Conda environment named `loco-ada`.

From the repository root, run:

```bash
conda env create -f environment.yml
conda activate loco-ada
```

The Python environment includes:

- `python=3.12`
- `numpy=1.26`
- `scipy=1.12`
- `matplotlib=3.8`

No extra package-install step is required after activating the Conda environment.

## Run the Python version

From the repository root:

```bash
conda activate loco-ada
python scripts/run_python_simulation.py
```

Useful options:

- `python scripts/run_python_simulation.py --no-plots`
- `python scripts/run_python_simulation.py --iterations 100`
- `python scripts/run_python_simulation.py --seed 42`
- `python scripts/run_python_simulation.py --split-or-tied tied --speed-protocol 'single speed change'`

## MATLAB setup

### Required MATLAB toolboxes

No additional MATLAB toolboxes are required.

The MATLAB code uses base MATLAB functionality only, including:

- `ode45`
- `interp1`
- standard plotting functions

### Run the MATLAB version

Open MATLAB, go to the `matlab/` folder, and run:

```matlab
run_simulation
```

This adds `matlab/src` to the path and runs the full MATLAB workflow.

## MATLAB/Python parity

The Python implementation has been checked against the MATLAB implementation using deterministic, no-noise reference runs.

To export a MATLAB reference:

```matlab
cd matlab/validation
export_reference_run('/tmp/locomotor_learning_model_reference.mat', 10)
```

To compare that reference against Python:

```bash
conda activate loco-ada
python scripts/validate_matlab_python_parity.py /tmp/locomotor_learning_model_reference.mat
```

Additional validation scripts:

```bash
conda activate loco-ada
python tests/test_matlab_python_equivalence.py
python tests/test_detailed_function_comparison.py
```

## Main parameter files

- Python protocol settings: `src/locomotor_learning_model/parameter_loading/load_protocol_parameters.py`
- Python learner settings: `src/locomotor_learning_model/parameter_loading/load_learner_parameters.py`
- MATLAB protocol settings: `matlab/src/loadProtocolParameters.m`
- MATLAB learner settings: `matlab/src/loadLearnerParameters.m`
