# Local validation

The subsequent [fresh-install test](fresh-install.md) passed **137 tests** in a new Conda environment, including the public-data workflow. The record below describes the preceding cleanup validation; its MATLAB and workbook checks were not repeated in the fresh-install test.

Validated October 15, 2026 in the Conda environment `loco-ada`: Python 3.12.13, NumPy 1.26.4, SciPy 1.17.1, Matplotlib 3.8.2 and pandas 3.0.1 on macOS arm64.

| Check | Result |
|---|---|
| Full tests with publication data | **94 passed**, no failures or skips |
| Static checks | Broad Ruff and whitespace checks passed |
| Scientific inputs | All **3,878** hashes match the unchanged data manifest |
| Figure workflow | **47 CSV tables and 42 figure files**; seven available workflows passed; Figures 1, 2 and S3 explicitly partial |
| Numerical cleanup comparison | 45 tables byte-identical; two nonlinear-fit diagnostic tables differ by at most **1.49×10⁻⁷**, below reported precision |
| Final manuscript comparison | 17 of 18 numerical claim groups match; low-speed inverse-fit RMSE remains 0.37148 versus printed 0.33 |
| New simulation commands | Demo completed 40 iterations; separate seeded pause/resume completed 3→8 iterations without replacing the source checkpoint |
| Built package | Wheel built and imported outside the repository; one-iteration demo saved results and checkpoint; deleted modules absent |
| MATLAB baseline plotting cleanup | Split and tied plot XData/YData identical before and after removing unused geometry calculations |
| Source workbook | 10 sheets, 13 tables, 14 source CSVs and 16,487 cells checked against current outputs |
| Independent S1 analysis, October 14 | MATLAB R2025a reproduced all three reference CSVs exactly; the MATLAB analysis source is unchanged |

The primary results are `outputs/publication/report.md` and `.json`. The final manuscript comparison is `outputs/manuscript-validation/report.md` and `.json`. `validation.json` records versions, checksums, numerical comparisons and cleanup details. `data/publication/manifest.json` is the single authoritative input manifest. The DATA download’s `validation/fall_fit_comparison.csv` preserves the independent MATLAB/SciPy fit check, and `final-manuscript-protocol-check.json` records the check of all 1,750 Figure 3 speed profiles.

## Cleanup

Removed the obsolete virtual environment, stale distribution packages, build/cache files, duplicate manifests, superseded generated reports and unused Python/MATLAB helpers. The redundant incoming handoff was removed after SHA-256 verification: 2,735 files already exist identically in the prepared dataset. Its 127 unique author files are preserved outside the repository at `../locomotor_learning_model_author_sources/2026-10-15`, with a manifest sufficient to reconstruct the original handoff. Earlier detailed audit notes are also retained there. None of these external files is required to run the prepared publication workflow.

Additional cached scientific trials, source metadata and separate model implementations remain because they document the study and support reuse. Simulation settings and the manuscript engine were preserved. The compact example checkpoints remain documented test fixtures.

## Scope

Software correctness does not establish complete manuscript agreement. See the [reproduction guide](../docs/reproducibility.md#coverage-and-limits) for coverage and limits. The numerical checks use saved scientific inputs; they do not certify unavailable raw-to-paper reconstruction or identical manual artwork. The continuous-integration workflow was updated but not run remotely during this validation session. No files were uploaded or published during that session.
