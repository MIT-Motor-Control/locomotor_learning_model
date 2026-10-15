# Fresh installation test — October 15, 2026

The README workflow was tested in a separate source copy and a newly created Conda environment on macOS arm64. The copy initially contained no prepared data, generated outputs or private author folders. The environment YAML installed successfully without changes. The test used `loco-ada-readme-test-20261015` instead of `loco-ada` to preserve the existing environment.

Computer Use refused access to Hyper for safety reasons, so execution took place in the workspace shell. An inherited Python-manager path initially selected an unrelated Python 2.7 installation after Conda activation. A clean shell selected the new environment correctly; all reported results below used its Python 3.12.13 with user-site packages disabled.

| Check | Result |
|---|---|
| Conda installation and dependency consistency | Passed; no extra scientific packages installed manually |
| README demo | Completed 40 learning iterations |
| README longer simulation | Completed 1,000 iterations / 2,000 physical steps, seed 42 |
| Supplied checkpoint | Resumed from iteration 10 to 20 |
| Experimental DATA import | Eight prepared files; original downloads unchanged |
| Experimental-only reproduction | Figures 1 and S3 partial; other eight workflows blocked; no execution failures |
| Companion ZIP installation | Upgraded the experimental dataset and verified all 3,878 inputs |
| Full available figure workflow | Seven passed; Figures 1, 2 and S3 partial; 14 plots in PNG/PDF/SVG and 47 CSV tables |
| Selected-figure command | Figure 3, S1 and S5 passed |
| Full test suite | **137 passed**, no failures or skips |
| Static checks and `pip check` | Passed |
| Built wheel outside the source checkout | Imports, two-iteration simulation, public-data import and experimental figures passed |

The full-input tables were compared with the preceding validated run. Forty of 47 were byte-identical. Seven had numerical differences of at most 5.23 × 10⁻⁹, below the reported precision. The scientific inputs and their manifest were unchanged. The experimental CSV route preserves six-decimal source rounding and does not claim to recover unavailable precision.

## Downloads and release status

The experimental data came from the user-supplied `locomotor_learning_model-DATA` folder, copied into the documented `data/` location. The additional model inputs came solely from a standalone `publication-data.zip`; the isolated workflow did not read author research or handoff directories.

`dist/publication-data.zip` is the local candidate for depositing alongside the experimental archive. Its SHA-256 is:

```text
aab4c976ef798efdc264a79479f2755c294c0576ed482e76b1508884c56d7717
```

The archive has not been uploaded by this test. The latest Zenodo page inspected still has restricted files, so an anonymous download-to-figure test remains a release requirement. This local result does not imply that the existing Zenodo DATA download alone contains the model inputs.

## Evidence and limits

`fresh-install.json` records environment versions, archive identity, input hashes, figure statuses and numerical comparisons. Local figures and reports are under `outputs/fresh-conda-validation/`; `figures.html` provides a gallery. Those generated outputs are excluded from Git.

This verifies the Python baseline simulation and available saved-data analyses. It does not rerun all stochastic model sweeps, raw C3D/COSMED processing, OpenSim optimization or the separate MATLAB S1 computation. MATLAB S1 and the optional Node workbook formatter are not dependencies of the README figure commands. Windows, Linux and remote CI were not run in this session. See the [reproduction guide](../docs/reproducibility.md#coverage-and-limits) for coverage and limits.
