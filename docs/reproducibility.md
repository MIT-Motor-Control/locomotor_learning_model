# Reproducing the paper

The publication workflow reads saved experimental measurements, fitted parameters and simulation results. It writes every output to a separate directory and checks input hashes before analysis. Once the matching Zenodo release is public, start with `python scripts/download_data.py`, which downloads the single DATA ZIP. It verifies and installs the matching DATA folder, including its embedded model-input ZIP, inside `data/`. Allow about 30 GB free for installation; [data setup](../data/README.md) covers interrupted transfers and manual downloads. Author source directories are not required by this workflow.

## Commands

From the repository root, after installation:

```sh
loco-publication prepare
loco-publication verify-data
loco-publication reproduce --strict
loco-publication reproduce --strict --figures figure3 s1 s4 --output outputs/selected
loco-publication recompute-s1 --matlab /path/to/matlab
```

`reproduce` runs the numerical workflows for Figures 1–4, S1–S5 and fit-quality summaries. With only an older experimental download, Figures 1 and S3 are partial and model-dependent analyses are blocked. Each run writes PNG/PDF/SVG figures, numerical CSV tables and `report.md`/`report.json`. Output formats depend on the figure. The report records input, source and output hashes and dependency versions. Reusing an existing managed output directory requires `--overwrite`.

Exit codes: **0** means the requested available workflows ran without unexpected errors; **1** means an integrity or execution failure; **2**, with `--strict`, means at least one requested analysis is partial or blocked. A successful exit without `--strict` does not mean every manuscript panel is complete.

## Inputs and provenance

| Input directory | Contents | Used for |
|---|---|---|
| `original/` | Published fit summaries, experimental measurements, Sánchez comparison, OpenSim outputs and original risk landscape | Figures 1, 4, S3A/C, S4, fit quality |
| `regenerated/` | Regenerated fits, trial-average grid, cross-validation tables, COSMED arrays and 1,150 selected simulation rollouts | Figure 2B, S1 and S3B |
| `fall_probability/` | Complete author fall-simulation files | Figure 3 |
| `extensions/s2/`, `extensions/s5/` | Supplementary numerical results | S2, S5 |

`manifest.json` identifies each input's origin and SHA-256 hash. The experimental import converts the six-decimal CSV values to the array layouts used by the analyses and records participant ordering; it does not invent missing precision or post-adaptation data. The companion archive preserves full-precision author arrays and must agree with the experimental download within its rounding tolerance.

During maintainer preparation of that companion, the missing standalone energy MAT file was reconstructed from `split_energy`, `tied1_energy` and `tied2_energy` embedded in the original fit file, without refitting. Omitted fall trials were recovered from an earlier local archive; overlapping files had to agree before merging. Those historical sources are described by provenance labels in the manifest, not runtime file dependencies.

Figure 3 requires every one of its 1,750 trials. A partial dataset is rejected rather than changing the probability denominator. The saved heatmap is the complete original 20 × 20 × 3 surface, not a newly simulated replacement.

## What the available inputs reproduce

The report distinguishes a numerical reconstruction from a replot of a saved reference. S1 defaults to plotting verified MATLAB participant-level cross-validation and grid-sensitivity results. `recompute-s1` reruns both analyses from the regenerated trial-average grid. An independent release check found exact numerical agreement between these fresh results and the cached tables. It needs MATLAB with the Statistics and Machine Learning Toolbox; ordinary `reproduce` does not. MATLAB's approximate signed-rank p-value and the plotting script's exact SciPy p-value are distinct outputs, with their methods labeled.

S2 and S5 use numerical results, not digitized images. S5 additionally refits all 750 cached cell means, reselects the models and recomputes all held-out predictions, noise-floor estimates and tests. Figure 4 and S4 use the saved original landscape. Fit-quality analyses retain the original published fit. There is no neural-network training checkpoint: this model learns controller parameters and an internal dynamics estimate. See [model.md](model.md) for reusable simulations and resumable learner checkpoints.

The original inverse fit selects the lowest RMSE across parameter cells and individual stochastic trials. S1 instead scores the average normalized trajectory of nonfalling trials within each cell, requiring at least five survivors. Both the realization and the fitting estimator differ; comparing their RMSE values directly would be misleading.

S2 uses 20 experimental participants (7/7/6 by condition) and five model seeds with shared learning rate 6×10⁻⁵, symmetry weight 1 and zero fall-penalty weight. These are the supplied shared Model A results, not the main participant-specific fits. Its reported STA test indicates no detected early-to-late difference, not proof of equivalence or zero adaptation. The manuscript's description of these predictions needs to identify this model variant.

Figure 2B follows the supplied regenerated-trajectory analysis at Δv = 0.7 m/s: smooth and normalize each trajectory, align it to exposure percentage, then average the nine participants and nine fitted simulations. Its exported tables distinguish raw group trajectories from their exponential fits. This reproduces the supplied panel; the manuscript caption's description of a representative participant needs clarification.

S3B reads exactly 50 regenerated trials at each of the 23 unique original best-fit parameter cells. It excludes falls, takes each participant's median energy change across surviving trials, and averages those medians by condition. Shared parameter cells are reused for their corresponding participants. The source table has 1,150 unique trials; participant-weighted counts in the condition summary are larger. The full 60,000-trial sweep is not needed to reconstruct this panel.

The `validate-manuscript` command checks freshly generated results against a historical manuscript snapshot, identified by PDF SHA-256 `6c73972316e0044884175e7399696c66621bd15f0f8b7d83e39776ef572784ae`. It verifies that exact PDF, inputs, analysis code and output hashes before comparison and rejects a different PDF:

```sh
loco-publication validate-manuscript \
  --manuscript /path/to/KangMitraSeethapathi_2026_FINAL.pdf \
  --results outputs/publication \
  --output outputs/manuscript-validation
```

Use an empty comparison-output directory. Exit code **2** means agreement is incomplete; **1** indicates an integrity or execution error. The generated report records the comparison and its limits. This snapshot predates the RMSE correction described below; the command does not validate later manuscript revisions. S1 annotates the manuscript's MATLAB p-value; the alternative SciPy value remains exported separately.

Figures have editable text and consistent rendering; manually composed schematics, arrows and the exact submitted page layout remain separate artwork. Newly generated stochastic simulations are not substitutes for the published saved results.

## Coverage and limits

- **Figure 1:** experimental observations, paired tests and the Sánchez comparison are reconstructed, with the time windows exported alongside source values.
- **Figure 2B:** the regenerated group-trajectory panel and its correlation are reconstructed. The caption's participant description remains unresolved.
- **Figures 3–4 and S1–S5:** the supplied saved-data analyses, Figure 4 parameter tests, S1 relaxed-grid checks and S3B rollouts are available. The Figure 4 parameter test is not a test of the separate fall-probability claim in the text.
- **S3A/OpenSim:** the five supplied optimization values are plotted. An independent calculation from the available muscle-output files is exported separately; those files do not establish all five published values. Running the original optimization from scratch is not supported by the supplied inputs.
- **RMSE correction:** the original 0.4 m/s cohort has mean inverse-fit RMSE 0.3715 across eight participants. The manuscript source reviewed on October 18, 2026 reports 0.37 ± 0.32, resolving the earlier 0.33 mean discrepancy at the reported precision. The historical PDF check retains its original target and result; no participant values or simulation parameters were changed.
- **Simulation protocol:** the supplied engine uses a 25-second transition and belt-speed division by 2; Methods state 20 seconds and division by √(g × leg length). These differences remain unresolved.
- **Simulation scope:** seeded examples and saved-result analyses are supported. Full original stochastic sweeps and raw COSMED/C3D processing are not reconstructed by this workflow. Saved-reference agreement does not establish agreement of every Methods statement.

Inspect each generated report for incomplete inputs and numerical comparisons before using its outputs.

## Testing

Figure 1 preserves MATLAB's midpoint-percentile box statistics, rather than the different default quartiles in Matplotlib. Exported box summaries make the plotted geometry inspectable. Figure 3D uses the same exponential model and unweighted least-squares objective as the source, with a portable SciPy solver. An independent base-MATLAB optimization agreed within 0.000003 strides at the observed speeds; byte-identical coefficients from the unavailable Curve Fitting Toolbox are not claimed.

```matlab
addpath('matlab/validation');
validate_fall_fit('outputs/publication/tables');
```

```sh
python -m pytest -q
```

Data-dependent tests skip when the prepared dataset is absent. A release check should prepare/extract the data, run the full tests, regenerate the figures, recompute S1 with MATLAB, and inspect the resulting figures. Use `--strict` for a specific figure only when all its inputs and analyses are expected to be complete.

## Editor source workbook

The requested 1C/D/E, 3C, S1 and S3 numerical values are exported in `outputs/publication/tables/`. They include the Sánchez comparison, participant observations, all 1,750 Figure 3 trials and all 1,150 S3B trials. These CSVs and the documented pipeline provide the reusable figure source data.

To gather the editor-requested source values into a workbook, use the same Conda environment:

```sh
python scripts/build_source_workbook.py outputs/publication outputs/source-data.xlsx
python scripts/check_source_workbook.py outputs/publication outputs/source-data.xlsx
```

The builder preserves the CSV values, records their hashes, and labels data sources and units. The checker compares workbook cells with the CSVs. Both use `openpyxl`, already included in the locked Conda environment; no extra runtime or private formatting package is needed.
