# Chain IFU corrections at two centers

This source-checked recipe is **unexecuted**; its data paths are placeholders.
It assumes an installed Hyphen environment with runtime dependencies active and
`hyphenate` on PATH. Single/batch IFU and frame merging are intended supported
workflows; MSA remains experimental. This batch does not merge cubes.

Supply four matched pairs of complete calibrated detector (CRF) and rectified
cube (s3d) FITS: frames `00001`, `00002`, each on `nrs1`, `nrs2`. Preserve WCS/ASDF,
data-quality and coverage extensions; extracted SCI arrays are insufficient.
No example dataset or installation recipe is supplied here.

From the checkout root, copy the [template](../examples/ifu_batch.yaml) and edit
input paths, centers and data-dependent science settings. Centers must be valid
and appropriate for every selected cube: the nested list `[[38,33],[40,35]]`
uses one-based `[x,y]` coordinates along FITS axes 1 and 2, selecting `SCI[:,32,37]` and
`SCI[:,34,39]`. Spatial radii are arcseconds assuming spatial CDELT in degrees;
spectral headers and WCS must use consistent units without automatic conversion.
Radii, smoothing, masks, fit orders and thresholds are illustrative choices.
Both thresholds are explicit: `-0.1` disables the null-fit test, while `0.5`
keeps fitting samples above half each spaxel's positive-weight median.
Quality failures remain for inspection.

```sh
cp examples/ifu_batch.yaml ifu-batch.local.yaml
# Edit ifu-batch.local.yaml before proceeding.
mkdir -p outputs
mkdir outputs/batch-01 && hyphenate ifu-batch.local.yaml
```

If `mkdir` refuses an existing `outputs/batch-01`, stop and choose a fresh path in both YAML and command.
Relative paths resolve from the launch working directory, even if the YAML moves
into `configs/`. `FRAME`, `DETECTOR` and `PIXEL` are literal, case-sensitive
substitutions. Keep all three in the output/plot prototypes and distinct centers
to avoid collisions; the parser does not establish collision safety for you.
Outputs/logs can overwrite, and disabling fitting is not a dry run.

The batch runs all four frame/detector pairs at `[38,33]`, then all four at
`[40,35]`: frame order first, detector order within each frame. The second center
reads the first center's output **for the same frame/detector**; it always reuses
that pair's original detector image. Overlapping selected regions can be fitted
again. For the second center on `00002/nrs2`, the unchanged template resolves to:

| Role | Path |
| --- | --- |
| Detector input | `data/detector/00002_nrs2_crf.fits.gz` |
| Cube input | `outputs/batch-01/00002_nrs2_38_33_hyphenated.fits.gz` |
| Cube output | `outputs/batch-01/00002_nrs2_40_35_hyphenated.fits.gz` |
| Log | `outputs/batch-01/00002_nrs2_40_35_hyphenated.log` |

With the shown recipe, expect eight basenames `FRAME_DETECTOR_PIXEL_hyphenated`,
each with `.fits.gz`,
`.log`, `.pdf`, `.pkl.gz.npz` and `.pkl.gz.json`: 40 files for a completed run.
The four `[38,33]` cubes are intermediate; the four `[40,35]` cubes are the final
chained products. Each initial IFU pass saves the pair even with plots disabled;
a save error stops before writing that pass's cube. The pair precedes retries.
`perform_dewiggling: false` loads it without rewriting it, but still writes cubes
and can refit with `mask_bad_fits: true`. Load only trusted artifacts because
loading permits pickle.

Inspect all products, log errors, fitted coverage, SCI shape/WCS, finite masks and
diagnostic spectra for both passes. With `mask_bad_fits: false`, quality failures
remain for inspection. IFU correction retains input ERR without propagating
fitted-model uncertainty. Completion, file counts and mocked routing tests do not
establish fit quality, spectral-feature preservation or scientific acceptance.
