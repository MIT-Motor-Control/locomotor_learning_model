# Reproducing the paper

The publication workflow reads saved experimental measurements, fitted parameters and simulation results. It writes every output to a separate directory and checks input hashes before analysis. Start with the [Zenodo download instructions](../data/README.md): place the matching DATA folder, including its embedded model-input ZIP, inside `data/`. Author source directories are not required by this workflow.

## Commands

From the repository root, after installation:

```sh
loco-publication prepare
loco-publication verify-data
loco-publication reproduce
loco-publication reproduce --figures figure3 s1 s4 --output outputs/selected
loco-publication recompute-s1 --matlab /path/to/matlab
```

`reproduce` runs all available workflows, including explicitly partial Figures 1, 2 and S3 with the companion archive. With only the experimental download, Figures 1 and S3 are partial and model-dependent analyses are blocked. Each run writes PNG/PDF/SVG figures, numerical CSV tables and `report.md`/`report.json`. Output formats depend on the figure. The report records input, source and output hashes and dependency versions. Reusing an existing managed output directory requires `--overwrite`.

Exit codes: **0** means the requested available workflows ran without unexpected errors; **1** means an integrity or execution failure; **2**, with `--strict`, means at least one requested analysis is partial or blocked. A successful exit without `--strict` does not mean every manuscript panel is complete.

## Inputs and provenance

| Input directory | Contents | Used for |
|---|---|---|
| `original/` | Published fit summaries, experimental measurements and original risk landscape | Main figures, S3 experimental/optimization panels, S4, fit quality |
| `regenerated/` | July 2026 trial-averaged fit grid and saved cross-validation results | S1 |
| `fall_probability/` | Complete author fall-simulation files | Figure 3 |
| `extensions/s2/`, `extensions/s5/` | Supplementary numerical results | S2, S5 |

`manifest.json` identifies each input's origin and SHA-256 hash. The experimental import converts the six-decimal CSV values to the array layouts used by the analyses and records participant ordering; it does not invent missing precision or post-adaptation data. The companion archive preserves full-precision author arrays and must agree with the experimental download within its rounding tolerance.

During maintainer preparation of that companion, the missing standalone energy MAT file was reconstructed from `split_energy`, `tied1_energy` and `tied2_energy` embedded in the original fit file, without refitting. Omitted fall trials were recovered from an earlier local archive; overlapping files had to agree before merging. Those historical sources are described by provenance labels in the manifest, not runtime file dependencies.

Figure 3 requires every one of its 1,750 trials. A partial dataset is rejected rather than changing the probability denominator. The saved heatmap is the complete original 20 × 20 × 3 surface, not a newly simulated replacement.

## What the available inputs reproduce

The report distinguishes a numerical reconstruction from a replot of a saved reference. S1 defaults to replotting the saved participant-level result; `recompute-s1` performs the original size-matched MATLAB analysis from the regenerated grid. MATLAB's approximate signed-rank p-value and the plotting script's exact SciPy p-value are distinct outputs, with their methods labeled.

S2 and S5 use numerical results, not digitized images. S5 additionally refits all 750 cached cell means, reselects the models and recomputes all held-out predictions, noise-floor estimates and tests. Figure 4 and S4 use the saved original landscape. Figure 2 and fit-quality analyses use the original saved fits, but the Figure 2 handoff example does not match the final PDF. There is no neural-network training checkpoint: this model learns controller parameters and an internal dynamics estimate. See [model.md](model.md) for reusable simulations and resumable learner checkpoints.

The `validate-manuscript` command checks freshly generated results against the numerical claims in the reviewed manuscript revision. It verifies the PDF, inputs, analysis code and output hashes before comparison:

```sh
loco-publication validate-manuscript \
  --manuscript /path/to/KangMitraSeethapathi_2026_FINAL.pdf \
  --results outputs/publication \
  --output outputs/manuscript-validation
```

Use an empty comparison-output directory. Exit code **2** means agreement is incomplete; **1** indicates an integrity or execution error. The generated report records the comparison and its limits. S1 annotates the manuscript's MATLAB p-value; the alternative SciPy value remains exported separately.

Figures have editable text and consistent rendering; manually composed schematics, arrows and the exact submitted page layout remain separate artwork. Newly generated stochastic simulations are not substitutes for the published saved results.

## Coverage and limits

- **Figures 1 and S3:** available experimental and supplied optimization values are reproduced. Comparison data and the S3B simulation set are incomplete.
- **Figure 2:** the supplied participant example can be regenerated, but it is not the example in the final manuscript.
- **Figures 3–4 and S1–S5:** available saved-data calculations are reproduced as described above; S3 remains partial. The additional Figure 4 significance comparison and S1 relaxed-grid analysis are not included.
- **Simulation scope:** seeded examples and saved-result analyses are supported. Full original simulation sweeps, raw COSMED processing and OpenSim optimization are not reconstructed by this workflow. The manuscript engine records its supplied settings; saved-reference agreement does not establish agreement of every Methods statement.

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

`outputs/source-data-review.xlsx` gathers the requested 1C/D/E, 3C, S1 and S3 source values. It includes participant observations, all 1,750 Figure 3 trials, units and the outstanding panel gaps. It is a review copy while the Sánchez comparison and S3B remain incomplete. The numerical CSVs are the primary reusable source files.

The optional workbook formatter is `scripts/build_source_workbook.mjs`; it requires Node and `@oai/artifact-tool`, as described in its usage header. The scientific pipeline does not depend on this formatter. Verify an exported workbook against its source CSVs and adjacent validation JSON with:

```sh
python scripts/check_source_workbook.py outputs/publication outputs/source-data-review.xlsx
```

The check uses the optional `openpyxl` dependency for reading only. The locked environment includes it.
