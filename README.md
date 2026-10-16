# Locomotor learning model

Code and analysis for **Safety prioritization shapes locomotor adaptation and performance**, by Inseung Kang, Kanishka Mitra and Nidhi Seethapathi.

[Paper preprint](https://www.biorxiv.org/content/10.64898/2026.03.02.709033v1) · [Zenodo archive (all versions)](https://doi.org/10.5281/zenodo.19547419) · [Model guide](docs/model.md) · [Figure and data guide](docs/reproducibility.md)

| I want to… | Start here |
|---|---|
| Try the model without downloading data | [Get started](#get-started) |
| Run a longer simulation or resume one | [Run a simulation](#run-a-simulation) |
| Recreate the paper's available numerical figures | [Reproduce the paper figures](#reproduce-the-paper-figures) |
| Understand the data files | [Data setup](data/README.md) and the README inside the DATA download |
| Find a file or function | [Repository map](#repository-map) |

## Get started

Install [Miniforge](https://github.com/conda-forge/miniforge#install) to get Conda. Download this repository using GitHub's **Code → Download ZIP** button and extract it, or clone it with Git:

```sh
git clone https://github.com/MIT-Motor-Control/locomotor_learning_model.git
cd locomotor_learning_model
```

Open a terminal in the repository folder—the one containing `environment.yml`—and run:

```sh
conda env create -f environment.yml
conda activate loco-ada
loco-simulate demo
```

The environment installs the package and its dependencies. The demo runs 40 learning iterations and writes `outputs/demo/split_belt_summary.png`. It needs no DATA download or MATLAB. This is a short installation check and may finish before the treadmill reaches split-belt walking.

Run the commands below from this same repository folder. In each new terminal, activate the environment with `conda activate loco-ada` first.

## Run a simulation

Run 1,000 learning iterations with a repeatable random seed:

```sh
loco-simulate run --seed 42 --iterations 1000 --output-dir outputs/my-run
```

| Output | What it contains |
|---|---|
| `split_belt_summary.png` | Simulation overview; tied walking produces `tied_belt_summary.png` |
| `run.npz` | Numerical results that can be loaded and replotted |
| `metadata.json` | Settings, seed, runtime versions and completed iterations |
| `checkpoint.npz` and `checkpoint.metadata.json` | Saved learner state and its description |

Use a different output folder to keep separate runs. To pause a short example halfway through and then continue it:

```sh
loco-simulate run --seed 42 --iterations 20 --stop-after 10 --output-dir outputs/paused
loco-simulate resume outputs/paused/checkpoint.npz --output-dir outputs/resumed
```

The resumed run finishes the original target of 20 iterations and preserves the input checkpoint. These examples need only the code. [The model guide](docs/model.md) explains simulation settings, checkpoint compatibility, the Python API and the separate manuscript MATLAB engine. Use `loco-simulate run --help` for all options.

## Reproduce the paper figures

The figure workflow needs the matching CODE and DATA release. GitHub contains code; the measurements, fitted models and saved simulation trials are distributed on [Zenodo](https://doi.org/10.5281/zenodo.23402153).

**1. Get the data.** Once the matching Zenodo files are public, run:

```sh
python scripts/download_data.py
```

The command downloads the complete DATA ZIP or its smaller pieces, checks the file hashes and extracts `data/locomotor_learning_model-DATA/`. Allow about **42 GB of free space** for download and installation. Rerun the command after an interruption to reuse downloaded data. An existing DATA folder is never overwritten.

If you already downloaded the DATA ZIP, extract its `locomotor_learning_model-DATA` folder into this repository's `data/` folder. Keep the nested `publication-data.zip` compressed. Your layout should be:

```text
locomotor_learning_model/
├── environment.yml
├── data/
│   └── locomotor_learning_model-DATA/
│       ├── README.md
│       ├── processed_csv/
│       ├── processed_mat/
│       ├── raw_mocap_c3d/
│       ├── subject_information/
│       ├── checkpoints/
│       └── publication-data.zip
└── …
```

**2. Prepare the inputs and make the figures.**

```sh
loco-publication prepare
loco-publication verify-data
loco-publication reproduce --strict
```

If the DATA folder is beside this repository instead, use this first command to avoid copying the large recordings:

```sh
loco-publication prepare --source ../locomotor_learning_model-DATA
```

Then run the same `verify-data` and `reproduce` commands. Preparation creates `data/publication/` and verifies all 5,166 scientific inputs. The standard figure workflow runs in Conda; MATLAB is only needed to rerun the optional MATLAB calculations.

**3. Open the results.**

| Location | Contents |
|---|---|
| `outputs/publication/report.md` | What ran, input coverage and scientific limitations |
| `outputs/publication/figures/` | Figures in PNG, PDF and SVG formats |
| `outputs/publication/tables/` | Numerical CSV source tables |

To repeat a run, use `loco-publication reproduce --strict --overwrite`. To select figures, use `loco-publication reproduce --figures figure3 s1 s5 --output outputs/selected`.

These commands reconstruct the **available numerical panels from saved measurements and simulations**. Figures 2B, S1 and S3B use the regenerated realization; other main analyses retain the original published fits. Fresh stochastic simulations need not produce identical trajectories. Original OpenSim optimization, full original sweeps, raw-recording preprocessing and manual figure artwork are outside this workflow. Remaining manuscript discrepancies are documented in the [reproduction guide](docs/reproducibility.md#coverage-and-limits).

See [data setup](data/README.md) for file formats, manual downloads and older experimental-only datasets.

## Repository map

```text
locomotor_learning_model/
├── README.md                 Installation and common commands
├── environment.yml           Conda environment
├── requirements-lock.txt     Pinned scientific dependencies
├── pyproject.toml            Package configuration and command definitions
├── CITATION.cff, LICENSE     Citation metadata and code license
├── src/locomotor_learning_model/
│   ├── learning/             Dynamics, controller learning and objectives
│   ├── initializing/         Initial conditions and treadmill state
│   ├── parameter_loading/    Model, learner and protocol settings
│   ├── post_processing/      Simulation results and plots
│   ├── publication/          Figure analyses and data verification
│   └── cli.py, workflow.py, artifacts.py  Commands, runs and checkpoints
├── matlab/                   MATLAB baseline, manuscript engine and validation
├── data/                     Downloaded DATA and prepared inputs; see its README
├── docs/                     Detailed model and reproduction guides
├── scripts/                  DATA downloader and source-workbook tools
├── tests/                    Numerical, checkpoint and data-integrity tests
├── reproducibility/          Recorded installation and scientific checks
├── .github/workflows/        Automated tests
└── outputs/                  Generated results; excluded from Git
```

## Check the installation

```sh
python -m pytest -q
```

Tests needing the publication data skip when it is absent. The [installation record](reproducibility/fresh-install.md) describes the tested environment and scope. Generated results belong in `outputs/`; data, checkpoints and release archives are excluded from Git.

Please cite the paper and the Zenodo version you use. Code is MIT licensed; experimental data are CC BY 4.0 and retain their source provenance. Questions: Kanishka Mitra, mitra819@mit.edu.
