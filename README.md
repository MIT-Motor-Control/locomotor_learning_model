# Locomotor learning model

Code for **Safety prioritization shapes locomotor adaptation and performance**, by Inseung Kang, Kanishka Mitra and Nidhi Seethapathi.

[Read the bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.03.02.709033v1) · [Data on Zenodo](https://doi.org/10.5281/zenodo.19547419)

Use this repository to run the locomotor learning model, resume a saved learner, or reproduce the available manuscript figures from saved data.

## Get started

Install [Miniforge](https://github.com/conda-forge/miniforge#install) if you do not already have Conda. Open a terminal in this repository—the folder containing this README—and run:

```sh
conda env create -f environment.yml
conda activate loco-ada
loco-simulate demo
```

The demo runs 40 learning iterations with seed 42 and saves its results in `outputs/demo/`. Open `split_belt_summary.png` to see the result. This is a short installation check; it may finish before the treadmill reaches split-belt walking. No publication dataset or MATLAB installation is needed for the Python demo.

The environment installs the package, testing tools and pinned scientific dependencies automatically. Keep this terminal's `loco-ada` environment active for the commands below.

## Run a simulation

Start a longer run:

```sh
loco-simulate run --seed 42 --iterations 1000
```

Results go to `outputs/run/`. A run saves:

| File | Contents |
|---|---|
| `split_belt_summary.png` | Overview of the simulation (`tied_belt_summary.png` for tied walking) |
| `run.npz` | Numerical results for later analysis or plotting |
| `metadata.json` | Parameters, seed, runtime versions and completed iterations |
| `checkpoint.npz` | Learner state for continuing the simulation |
| `checkpoint.metadata.json` | Readable description of the saved learner state |

To create a checkpoint and continue it, pause a short run halfway through:

```sh
loco-simulate run --seed 42 --iterations 20 --stop-after 10 --output-dir outputs/example
loco-simulate resume outputs/example/checkpoint.npz
```

The second command continues from iteration 10 to the saved target of 20 and writes to `outputs/resumed/`, preserving the input checkpoint. These commands need only the code. Optional example checkpoints are included in the DATA release; they demonstrate the interface and are separate from the published fitted parameters.

Use `--output-dir outputs/my-run` to keep a separate run; repeating a simulation command writes to the same default output folder. Add `--show-plots` to open interactive figures. For all options:

```sh
loco-simulate run --help
loco-simulate resume --help
```

The [model guide](docs/model.md) covers pausing, extending runs, tied-belt protocols, the Python API and numerical reproducibility. The [manuscript MATLAB engine](matlab/manuscript/README.md) has a separate guide because its scientific settings differ from the Python baseline.

## Reproduce the paper figures

Use the DATA archive accompanying the same code release on [Zenodo](https://doi.org/10.5281/zenodo.19547419). Download `locomotor_learning_model-DATA.zip`, extract it, and put the extracted folder inside this repository's `data/` folder. The matching DATA release includes the experimental measurements, example checkpoints, and a nested `publication-data.zip` containing fitted models and saved simulation trials. Keep that inner ZIP compressed; preparation handles it automatically. GitHub contains the code, documentation and tests; the release data are distributed in the DATA ZIP.

Your folders should look like this:

```text
data/
└── locomotor_learning_model-DATA/
    ├── processed_csv/
    ├── processed_mat/
    ├── raw_mocap_c3d/
    ├── subject_information/
    ├── checkpoints/                  Optional saved simulation examples
    ├── publication-data.zip          Fitted models and saved simulation trials
    └── README.md
```

An older experimental-only DATA download supports the available experimental panels in Figures 1 and S3 but lacks model fits and simulation sweeps. Use the matching DATA release for the model analyses. [Data setup](data/README.md) explains both cases.

The updated DATA archive is prepared for release. Its upload to the existing Zenodo record is pending correction of that record's DOI/versioning configuration; the currently published DATA download is the older experimental-only version.

```sh
loco-publication prepare
loco-publication verify-data
loco-publication reproduce
```

Preparation creates `data/publication/`, checks the downloaded inputs, and records their provenance. It uses only the folders above. Without the companion archive, model-dependent analyses are reported as blocked; nothing is silently fetched from an author's computer.

Open `outputs/publication/report.md`. Figures are in `outputs/publication/figures/`; numerical source tables are in `outputs/publication/tables/`. This saved-data workflow needs only the Conda environment, not MATLAB or Node. To repeat an existing run, add `--overwrite`. To generate only selected figures:

```sh
loco-publication reproduce --figures figure3 s1 s5 --output outputs/selected
```

Figures 1, 2 and S3 are partial with the available inputs. A successful plotting run verifies the available calculations, not every manuscript claim. See [reproduction details](docs/reproducibility.md) for coverage, source data, MATLAB cross-validation and manuscript checks.

## Repository map

```text
locomotor_learning_model/
├── README.md                         Start here
├── environment.yml                   Conda setup
├── requirements-lock.txt             Exact scientific dependency versions
├── pyproject.toml                    Package metadata and command definitions
├── MANIFEST.in                       Files included in a source release
├── CITATION.cff                      Software citation metadata
├── LICENSE                           Code license
├── src/locomotor_learning_model/
│   ├── cli.py, __main__.py            Simulation commands
│   ├── workflow.py, artifacts.py      Run orchestration and checkpoints
│   ├── initializing/                 Initial physical/controller states
│   ├── parameter_loading/            Model and treadmill settings
│   ├── learning/                     Dynamics, learning and objectives
│   ├── post_processing/              Results and plots
│   └── publication/                  Figure analyses, data preparation and audits
├── matlab/
│   ├── run_simulation.m, src/         MATLAB baseline for Python comparisons
│   ├── manuscript/                   Separate manuscript engine and seeded runs
│   └── validation/                   Numerical comparisons
├── data/
│   ├── README.md                     Dataset setup and provenance
│   ├── locomotor_learning_model-DATA/  Extracted DATA release, including model inputs and checkpoints
│   └── publication/                  Prepared inputs; created by prepare
├── docs/
│   ├── model.md                      Simulation guide
│   └── reproducibility.md            Figure workflow and source-data guide
├── scripts/
│   ├── validate_matlab_python_parity.py  Compare the two baseline implementations
│   ├── build_source_workbook.mjs      Format the optional editor source workbook
│   └── check_source_workbook.py       Check workbook values against source CSVs
├── tests/                            Numerical, checkpoint and integrity tests
├── reproducibility/
│   ├── validation.md, validation.json  Cleanup checks and record
│   ├── fresh-install.md, fresh-install.json  Clean Conda installation test
│   └── final-manuscript-protocol-check.json  Cached treadmill-protocol check
├── .github/workflows/                Automated software tests
└── outputs/                          Generated runs and figures; excluded from Git
```

Each run belongs in `outputs/`, not among the source files. `dist/` is created only when packaging a release. The complete scientific dataset retains additional archived trials and original source formats for provenance and reuse; it is kept separate from the code.

## Check the installation

```sh
python -m pytest -q
```

Data-dependent tests skip if the publication dataset is absent. For a shell without activation, prefix commands with `conda run --no-capture-output -n loco-ada`. If another Python manager overrides Conda, check that `python -c "import sys; print(sys.executable)"` points inside `envs/loco-ada`; deactivate stacked environments and activate `loco-ada` again if needed.

The clean-install test and its limits are recorded in [fresh-install validation](reproducibility/fresh-install.md).

Please cite the preprint and the archived version you use. Code is MIT licensed; data retain their own license and provenance. Questions: Kanishka Mitra, mitra819@mit.edu.
