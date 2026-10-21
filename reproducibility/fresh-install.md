# Installation and reproduction checks

The README was verified on October 18, 2026 in a separate code copy and a fresh Conda environment on macOS arm64. The copy started without prepared data, generated outputs or author source folders. Installation used `environment.yml`; no additional packages were installed manually. The source cleanup was checked on October 21 using the same pinned scientific dependencies; the results below distinguish installation checks from that regression run.

## Verified workflows

| Check | Result | Last checked |
|---|---|---|
| Conda installation | Python 3.12.13; dependency consistency check passed | October 18 |
| README demo | 40 learning iterations completed | October 18 |
| Longer seeded simulation | All numerical outputs matched commit `040a307` exactly across 1,000 iterations | October 21 |
| Checkpoint continuation | Pause/resume and uninterrupted runs agreed in regression tests | October 21 |
| Local DATA ZIP installation | Checksum verified; all 105 data/supporting files installed | October 18 |
| Prepared inputs | All 5,166 scientific inputs verified | October 21 |
| Figure reproduction | All ten workflows passed in strict mode; 45 figure files and 61 CSV tables | October 21 |
| Numerical comparison | All 61 CSV tables were byte-identical to the previous reconstruction | October 21 |
| Test suite | 287 tests passed without skips | October 21 |
| Lint and dependency checks | Passed | October 21 |

## DATA archive

The verified single DATA ZIP contains **13,269,924,017 bytes**, with SHA-256:

```text
7fe389e9125e56338a5e99bd1448474ef20447930fc1aaaf3c3becb02679c945
```

Use the README's `--archive` command to install an existing download. On October 18 the public download returned HTTP 404 because the matching Zenodo version was unpublished. Anonymous download access requires a separate check after publication.

## Additional validation

MATLAB R2025a checks on October 21 covered seeded repeatability, saved-trial reuse, setting-mismatch rejection, sweep restart, fall-stride recording and odd/even step budgets. Fresh MATLAB S1 calculations passed. A deterministic 40-iteration baseline MATLAB/Python comparison agreed within 5.8 × 10⁻¹² for the objective and 5.1 × 10⁻¹⁶ for controller histories. The deterministic export completed with rank-deficient rolling-estimator warnings.

The editor source workbook passed 29,397 cell comparisons across 25 CSV tables and 15 sheets on October 21. A wheel built from the cleaned source imported outside the checkout, included both MATLAB S1 analysis files and completed a two-iteration simulation.

Dated machine-readable records retain the archive identities, numerical comparisons and environments used for those checks:

- [`fresh-install.json`](fresh-install.json): initial installation and table comparisons.
- [`release-check.json`](release-check.json): follow-up software and archive checks.
- [`final-manuscript-protocol-check.json`](final-manuscript-protocol-check.json): all 1,750 cached Figure 3 speed profiles.

These earlier records identify the archives tested on their respective dates; the single ZIP above is the current download.

## Scope

These checks cover macOS arm64, the baseline simulation and reconstruction from saved numerical inputs. They do not rerun the full original stochastic sweeps, raw C3D/COSMED preprocessing, OpenSim optimization or manually composed artwork. Windows validation is pending. MATLAB is required for the explicitly documented MATLAB workflows.

The historical PDF checker targets a revision that predates the corrected 0.4 m/s RMSE. Its recorded 19-of-20 comparison is specific to that PDF. The manuscript source reviewed on October 18 reports 0.37 ± 0.32, matching the eight-participant calculation; the October 19 Methods corrections match the manuscript engine's 25-second transition and belt-speed scaling. See the [reproduction guide](../docs/reproducibility.md#coverage-and-limits) for remaining source and interpretation limits.
