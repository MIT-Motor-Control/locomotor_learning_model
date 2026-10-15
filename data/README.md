# Publication data

From the repository root, download and install the matching [Zenodo DATA release](https://doi.org/10.5281/zenodo.23402153):

```sh
python scripts/download_data.py
```

The release distributes the original 13,335,231,738-byte `locomotor_learning_model-DATA.zip` as 54 pieces: the first 53 are 250,000,000 bytes each and the final piece is 85,231,738 bytes. The downloader checks the pinned manifest and every piece, reconstructs the original ZIP, verifies its complete SHA-256, and safely extracts `data/locomotor_learning_model-DATA/`. No extra download package is needed.

Allow about **42 GB free** while downloading, joining and extracting; the installed DATA folder occupies about 14.88 GB before publication preparation. Verified pieces are cached in `outputs/data-download/23402153/`. Rerun the same command after an interruption: completed pieces are reused, while an incomplete piece restarts. Cache files created by the command are removed after a successful installation; use `--keep-downloads` to retain them. Existing datasets and unrelated cache files are never overwritten or removed.

If you downloaded the pieces manually, place all 54 `locomotor_learning_model-DATA.zip.part001` through `.part054` files and `locomotor_learning_model-DATA.manifest.json` in one directory, then run:

```sh
python scripts/download_data.py --parts-dir /path/to/downloaded-pieces
```

This mode makes no network requests and preserves the supplied pieces. The downloader refuses an existing installation folder. Use the existing dataset with `prepare`, or install separately and supply that location:

```sh
python scripts/download_data.py --destination data/another-release
loco-publication prepare --source data/another-release
```

The original ZIP's SHA-256 is `14e9e46710afffd3998d0f5a5f33933c42d5599f2856eb743e1f025f5ac17cea`. The manifest and archive identity are pinned in the script; use code and data from the same release. Keep the extracted README and metadata with the data.

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
| `locomotor_learning_model-DATA.zip.part001`–`.part054` and the download manifest | Reassemble the unchanged DATA ZIP: experimental CSVs, processed MAT files, raw C3D recordings, metadata, checkpoints and the embedded model-input ZIP | All available saved-data analyses, with known gaps explicitly reported |
| Embedded `publication-data.zip` | Original and regenerated fits, cross-validation and grid-sensitivity tables, fall trials, Sánchez data, OpenSim outputs and supplementary model caches | Adds the model figures and analyses, including regenerated Figure 2B and S3B |

The experimental CSVs contain values rounded to six decimals. Their reconstructed arrays preserve that precision. The companion includes the original full-precision arrays; preparation checks their agreement with the CSVs within the rounding tolerance and retains the original arrays. Small last-decimal differences between experimental-only and full-input runs are expected. Neither route reruns raw C3D processing, COSMED preprocessing or the full stochastic simulation sweeps.

The raw C3D and per-participant MAT files remain available for reuse, but the figure pipeline reads the group-level CSVs. It does not need a C3D reader. `prepare` does not duplicate the large raw recordings. Experimental-only preparation reports the missing model inputs rather than substituting new simulations or invented values.

Eight double-support-asymmetry records have a missing final sample. These endpoints are preserved and listed in the import manifest; the importer does not fill or interpolate them. The energy and step-length-asymmetry inputs used for Figures 1 and S3 are complete.

## Access and versions

Use the CODE and DATA archives from the same Zenodo release. Earlier experimental-only DATA downloads do not contain the model fits or simulation caches; they support only the available experimental panels. The new DATA package includes those additional inputs and their checksum manifest.

Download availability is controlled by the [Zenodo record](https://doi.org/10.5281/zenodo.23402153). The downloader uses public file URLs and does not handle restricted-record credentials. The repository's MIT license covers code; the experimental data retain their CC BY 4.0 license. Retain the downloaded metadata and provenance when redistributing the data.

Release 0.3.0 adds the supplementary source package dated October 15, 2026. Figure 2B, S1 and S3B use the regenerated realization; other main analyses retain the original published fits. The manifest records these sources separately. Fresh stochastic sweeps are expected to agree statistically, not trajectory by trajectory.

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

For maintainers extending an existing prepared dataset with the supplementary source package:

```sh
loco-publication import-addendum --source /path/to/kanishka_addendum_2026-10-15 \
  --previous-bundle /path/to/InvLocAd_PLOSBio_figure_code
```

The importer verifies existing hashes and rejects conflicting numerical inputs. MATLAB grid-sensitivity tables may be added with `--generated-s1 /path/to/recomputed/tables` and `--generated-s1-script /path/to/publication_s1_grid_sensitivity.m`; their input and source hashes are preserved in the manifest. This is a release-preparation step, not a reader installation step.
