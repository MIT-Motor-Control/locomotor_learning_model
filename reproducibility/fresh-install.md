# Release 0.3.0 installation check — October 15, 2026

The README workflow passed in a new Conda environment on macOS arm64, using a separate code copy and the actual release DATA ZIP. The copy started without prepared data, outputs or author source folders. No additional scientific packages were installed manually. The test used the environment name `loco-ada-addendum-test-20261015` to preserve the existing installation; the environment YAML was unchanged.

| Check | Result |
|---|---|
| Conda installation and dependency consistency | Passed; Python 3.12.13 and the pinned scientific stack |
| README demo | Completed 40 learning iterations |
| Longer seeded simulation | Completed 1,000 iterations / 2,000 physical steps |
| Pause and resume | Continued from iteration 10 to 20, preserving the input checkpoint |
| DATA archive | All 105 ZIP members and 104 payload hashes checked; extracted into the documented location |
| Data preparation | All 5,166 scientific inputs verified |
| Full figure workflow, strict mode | All ten workflows passed; 15 plots in PNG/PDF/SVG and 61 CSV tables |
| Selected figures | Figure 3, S1 and S5 passed |
| Full test suite | **174 passed**, no failures or skips |
| Lint, whitespace and dependency checks | Passed |
| MATLAB manuscript engine | Seed repeatability, completed-trial reuse, setting-mismatch rejection and sweep restart passed |
| S1 MATLAB recomputation | LOSO and three grid-sensitivity summaries, including all 78 sensitivity folds, exactly matched cached numerical values |
| Editor source workbook | 15 sheets, 25 CSV tables and 29,397 source cells verified using the standard Conda environment |
| Built wheel | Imported outside the source checkout; bundled MATLAB source present; two-iteration simulation passed |

Across the isolated run and working-copy run, 51 of 61 numerical tables were byte-identical. Differences in the remaining ten tables were at most **5.23 × 10⁻⁹**, below the reported precision. The scientific input manifest was identical. Independent comparisons with the supplied scripts also matched all Figure 2B source metrics and all 26 S3B participant medians; the Sánchez values agreed with MATLAB within 1.14 × 10⁻¹³.

The figure workflow includes original published fits and the separate regenerated realization used for Figures 2B, S1 and S3B. It does not substitute newly simulated trajectories for saved manuscript results.

## Archive used for the initial installation check

The initial check used a DATA archive with 13,335,231,738 bytes and SHA-256:

```text
14e9e46710afffd3998d0f5a5f33933c42d5599f2856eb743e1f025f5ac17cea
```

Its embedded `publication-data.zip` has 1,352,215,131 bytes and SHA-256:

```text
7c9d64b8140db3d5f67781301036c0b2cd8a6ece734b6e3bde5597aad5c13b90
```

The inputs were supplied solely by that archive in the isolated test. No external author directories were used. Download availability and file access are controlled by the [Zenodo record](https://doi.org/10.5281/zenodo.19547419); this local installation test alone does not establish anonymous download access.

## Final code and DATA refresh

The final cleanup was checked in the working installation. **229 tests passed**, including six step-budget regression cases; lint and dependency checks passed. A 40-iteration MATLAB/Python comparison agreed within 5.8 × 10⁻¹² for the objective and 5.1 × 10⁻¹⁶ for controller histories. Actual MATLAB checks covered both default and odd step-count configurations. The demo, pause/resume commands and the example checkpoint from the standalone DATA folder ran successfully.

Preparing the neighboring DATA folder into an empty destination verified all 5,166 inputs without author source packages. All ten figure workflows passed again: 15 plots, 45 figure files and 61 CSV tables. Of those tables, 59 were byte-identical to the previous reconstruction; two differed only by floating-point rounding, at most 1.02 × 10⁻¹⁰. The rebuilt source workbook passed all 29,397 cell comparisons.

The final DATA ZIP is **13,335,229,428 bytes**, with SHA-256:

```text
7fcd2f925102bb8a2231932ab79d3bebaee9b92e32c3917150d52d09294424d6
```

Only the DATA README and its checksum index changed from the initially tested archive; all 103 scientific/supporting payload files are unchanged. All 105 final ZIP members were checked against the standalone folder, and all 54 optional download pieces were independently verified against the complete archive. The downloader pins this final version. [`release-check.json`](release-check.json) records these checks and the current archive identities; the earlier fresh-install record is retained unchanged.

## Evidence and limits

[`fresh-install.json`](fresh-install.json) records versions, the initially tested archive identity, input hashes, figure statuses and all numerical table comparisons. Generated outputs are excluded from Git. [`final-manuscript-protocol-check.json`](final-manuscript-protocol-check.json) retains the check of all 1,750 cached Figure 3 speed profiles.

The manuscript comparison matches **19 of 20 numerical claim groups**. The low-speed inverse-fit mean RMSE remains 0.3715 versus the printed 0.33. Separate source and interpretation issues also remain, including the group-average Figure 2B caption, supplied OpenSim values, Figure 4 risk significance, S2 model provenance and simulation settings. See the [reproduction guide](../docs/reproducibility.md#coverage-and-limits).

This verifies the software and available saved-data reconstruction. It does not rerun the full original stochastic sweeps, raw C3D/COSMED processing or OpenSim optimization, and does not certify manually assembled artwork. Windows was not tested in this session. MATLAB is only needed for the explicit MATLAB workflows; no Node or private formatting package is needed for figures or the source workbook.
