# Launch one IFU case

This source-checked recipe is **unexecuted**: its data paths are placeholders.
It assumes Hyphen and its runtime dependencies are installed, your chosen
environment is active, and `hyphenate` is on PATH. It does not establish an
installation recipe, Python/platform support or scientific acceptance.

Supply complete, matched calibrated detector (CRF) and rectified cube (s3d) FITS
for the same exposure/detector, including WCS/ASDF, data-quality and coverage
extensions. Extracted SCI arrays are insufficient; no example dataset is supplied.

From the checkout root, copy the [template](../examples/ifu_single.yaml), then edit
both input paths, the center and all data-dependent science settings. The shown
`central_pixel: [38, 33]` is one-based `[x,y] = [NAXIS1,NAXIS2]`, selecting NumPy
`SCI[:,32,37]`. Choose a valid center for your cube. Radii are arcseconds assuming
spatial CDELT values in degrees; spectral headers and WCS must have consistent
units without automatic conversion. The template disables the null-fit test and
retains quality failures; its radii, smoothing, fit orders and thresholds are
illustrative choices, not recommended defaults.

```sh
cp examples/ifu_single.yaml ifu.local.yaml
# Edit ifu.local.yaml before proceeding.
mkdir -p outputs
mkdir outputs/run-01 && hyphenate ifu.local.yaml
```

If `outputs/run-01` already exists, stop and choose a fresh path in the YAML and
command. Outputs/logs can overwrite. Relative paths resolve from the launch
working directory, even if the YAML is moved into `configs/`. `FRAME` and
`DETECTOR` are literal, case-sensitive filename substitutions.

For the shown frame/detector and unchanged output paths, expect
`outputs/run-01/00001_nrs1_hyphenated` with suffixes `.fits.gz`, `.log`, `.pdf`,
`.pkl.gz.npz` and `.pkl.gz.json`. Initial IFU fits save the pair even with plots
disabled; a save error stops before the final cube write. The pair captures the
initial pass, before retries. `perform_dewiggling: false` loads it without
rewriting it, but still writes the cube and can refit with `mask_bad_fits: true`;
it is not a dry run. Load only trusted artifacts because loading permits pickle.

Check all expected products, log errors, fitted coverage, SCI shape/WCS, finite
masks and diagnostic spectra. With `mask_bad_fits: false`, quality failures remain
for inspection. IFU correction retains input ERR without propagating fitted-model
uncertainty. A completion message, output file or readable guide alone does not
establish a successful fit, spectral-feature preservation or scientific acceptance.
