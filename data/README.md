# Publication data

Download `locomotor_learning_model-DATA.zip` from [Zenodo](https://doi.org/10.5281/zenodo.19547419), extract it, and place the resulting `locomotor_learning_model-DATA/` folder here. Its `processed_csv/` directory should be at `data/locomotor_learning_model-DATA/processed_csv/`, not one extra level deeper. Keep the downloaded README and metadata with the data.

The matching release includes `publication-data.zip` inside the extracted DATA folder. Leave that embedded archive compressed; preparation finds it automatically. The original experimental-only download can still be used with a separately supplied companion ZIP. Run these commands from the repository root with the Conda environment active:

```sh
loco-publication prepare
loco-publication verify-data
loco-publication reproduce
```

`prepare` creates `data/publication/`. It validates participant identities, conditions, time-series dimensions and numerical values before importing the six experimental CSVs. When the embedded or separately supplied companion ZIP is present, it checks its manifest and compatibility with the experimental download before installing the full prepared dataset. It refuses conflicting inputs. Repeating preparation with the same inputs is safe.

For another location, provide it explicitly:

```sh
loco-publication prepare --source /path/to/locomotor_learning_model-DATA \
  --model-inputs /path/to/publication-data.zip
```

No private directories, author handoff, MATLAB installation or network connection are needed after downloading the inputs. `data/publication/manifest.json` lists prepared inputs, checksums and transformations. The original downloads stay unchanged. After successful preparation, the companion ZIP may be removed locally to save space; keep the Zenodo version and checksum with your results.

## What each download supports

| Download | Contents | Reproduction coverage |
|---|---|---|
| `locomotor_learning_model-DATA.zip` | Experimental CSVs, processed MAT files, raw C3D recordings, metadata, checkpoints and the embedded model-input ZIP | All available saved-data analyses, with known gaps explicitly reported |
| Embedded `publication-data.zip` | Original and regenerated fits, cross-validation tables, fall trials, risk landscapes and supplementary model caches | Adds the available model figures and analyses; unresolved panels remain explicitly partial |

The experimental CSVs contain values rounded to six decimals. Their reconstructed arrays preserve that precision. The companion includes the original full-precision arrays; preparation checks their agreement with the CSVs within the rounding tolerance and retains the original arrays. Small last-decimal differences between experimental-only and full-input runs are expected. Neither route reruns raw C3D processing, COSMED preprocessing or the full stochastic simulation sweeps.

The raw C3D and per-participant MAT files remain available for reuse, but the figure pipeline reads the group-level CSVs. It does not need a C3D reader. `prepare` does not duplicate the large raw recordings. Experimental-only preparation reports the missing model inputs rather than substituting new simulations or invented values.

Eight double-support-asymmetry records have a missing final sample. These endpoints are preserved and listed in the import manifest; the importer does not fill or interpolate them. The energy and step-length-asymmetry inputs used for Figures 1 and S3 are complete.

## Access and versions

Use the CODE and DATA archives from the same Zenodo release. Earlier experimental-only DATA downloads do not contain the model fits or simulation caches; they support only the available experimental panels. The new DATA package includes those additional inputs and their checksum manifest.

Download availability is controlled by the [Zenodo record](https://doi.org/10.5281/zenodo.19547419). Restricted files require access from the record owner. The repository's MIT license covers code; the experimental data record uses CC BY 4.0. Retain the downloaded metadata and provenance when redistributing the data.

The updated DATA archive is prepared locally. Uploading it to the existing record is currently blocked: the published record uses the preprint DOI in its own DOI field, which disables new versions, and published file edits are locked after 30 days. The record's DOI/versioning configuration needs correction before this updated archive can be deposited there.

## Maintainers

To package the verified model inputs already in `data/publication/`:

```sh
loco-publication package-data
```

This writes `dist/publication-data.zip` and `dist/publication-data.zip.sha256`. Include both files inside `locomotor_learning_model-DATA/` before creating the DATA release ZIP. The model-input ZIP preserves the complete per-file manifest. To assemble a dataset from an original handoff, the older maintainer import remains available:

```sh
loco-publication prepare --bundle /path/to/InvLocAd_PLOSBio_figure_code \
  --fall-archive "/path/to/earlier/original_data/data/fall probability" \
  --private-archive /path/to/locomotor_learning_model_private
```

These maintainer sources are not dependencies for readers using the released archives. The prepared dataset keeps original and regenerated results distinct and retains additional archived trials for reuse. See the [reproduction guide](../docs/reproducibility.md#coverage-and-limits) for coverage and limits.
