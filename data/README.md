# Data setup

Use the CODE and DATA archives from the same [Zenodo release](https://doi.org/10.5281/zenodo.23402153). The DATA folder has its own README with file formats, units, participant groups, missing values and Python/MATLAB examples.

## Download and prepare

With the repository's Conda environment active, run:

```sh
python scripts/download_data.py
loco-publication prepare
loco-publication verify-data
loco-publication reproduce --strict
```

The downloader retrieves the single `locomotor_learning_model-DATA.zip` from Zenodo and verifies its pinned checksum before extracting into `data/locomotor_learning_model-DATA/`. The ZIP is about 13.27 GB; allow **30 GB free** for download and extraction. The extracted folder occupies about 14.88 GB before analysis preparation.

If you download the ZIP manually, extract its DATA folder into this repository's `data/` directory. Keep the inner `publication-data.zip` compressed. Then start with `loco-publication prepare` above.

If the DATA folder is beside the repository, no copy is needed:

```sh
loco-publication prepare --source ../locomotor_learning_model-DATA
loco-publication verify-data
loco-publication reproduce --strict
```

Preparation creates `data/publication/`, verifies the companion archive and checks its agreement with the experimental CSVs. All 5,166 prepared scientific inputs are recorded in `data/publication/manifest.json`. Raw recordings stay in the DATA folder and are not duplicated. Repeating preparation with the same inputs is safe; conflicting inputs are rejected.

## What is in the DATA folder?

| Path | Format and use |
|---|---|
| `processed_csv/` | Six CSV tables with one row per participant and columns for condition and time/stride samples; the experimental figure inputs |
| `processed_mat/` | Per-participant MATLAB measurements for additional analysis |
| `raw_mocap_c3d/` | Original motion-capture recordings; a C3D reader is needed for new raw-data analysis |
| `subject_information/` | Participant characteristics and condition assignments |
| `publication-data.zip` | Original and regenerated model fits, saved trials, cross-validation tables, comparison data and supplementary results |
| `checkpoints/` | Seeded model examples and metadata; these are separate from the fitted manuscript models |
| `validation/fall_fit_comparison.csv` | Independent MATLAB/SciPy comparison of the fall-adaptation fit |
| `SHA256SUMS` | Checksums for every other file in the DATA folder |

The standard Python figure workflow needs the six CSVs and embedded publication archive. It does not require MATLAB, a C3D reader or a network connection after downloading the data. MATLAB is needed only for the explicit MATLAB simulation and S1 recomputation commands.

## Interrupted downloads and other locations

Rerun `python scripts/download_data.py` after an interruption. Downloads resume when the server supports HTTP Range requests; otherwise they restart safely. Connection errors and checksum mismatches do not silently select a different dataset.

Downloads are cached in `outputs/data-download/23402153/` and removed after a successful installation. Add `--keep-downloads` to retain them. The downloader never overwrites an existing DATA folder. To install separately:

```sh
python scripts/download_data.py --destination data/another-release
loco-publication prepare --source data/another-release
```

## Precision and coverage

The experimental CSVs are rounded to six decimals. The publication archive retains the full-precision arrays; preparation checks agreement within the CSV rounding tolerance. Eight double-support-asymmetry records lack a final sample. Those endpoints remain missing and are recorded in the import manifest. Energy and step-length-asymmetry inputs used by the available figure analyses are complete.

Figures 2B, S1 and S3B use the regenerated simulation realization. Other main analyses retain the original published fits. Their provenance is recorded separately. A fresh stochastic sweep need not produce the same trajectories.

The release contains all inputs used by the available saved-data workflows. It does not reconstruct raw C3D/COSMED processing, the complete original stochastic sweeps or the original OpenSim optimization. Some manuscript numbers and descriptions remain unresolved; see [coverage and limits](../docs/reproducibility.md#coverage-and-limits).

Older experimental-only DATA releases lack `publication-data.zip` and support only part of Figures 1 and S3. Preparation reports missing model inputs rather than substituting new results. If you have a separate matching publication archive, use `loco-publication prepare --source /path/to/DATA --model-inputs /path/to/publication-data.zip`.

Experimental data are CC BY 4.0; code is MIT licensed. Retain the dataset README, checksums and source provenance, and cite the paper and the Zenodo version used.
