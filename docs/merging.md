# Merge four corrected cubes

This [template](../examples/merge.yaml) is **unexecuted**. Scientific use requires
the pending merger ERR/cache correction, human review and real-data validation.
The baseline incorrectly combines standard-deviation ERR as variance; the pending
correction uses independent-input variances and writes standard-deviation ERR.

Requires an installed `hyphenate` CLI and four trusted, complete corrected FITS
cubes: `00001`/`00002` crossed with `nrs1`/`nrs2`, named as the template expands.
Replace literal `DATA_ROOT` with their absolute folder; it is not an environment variable.
Data are not bundled or publicly hosted here. Do not use SCI extracts or uncorrected cubes.
Inputs must share shape and aligned spatial/wavelength grids; no resampling occurs.
HDU positions 1–4 must be SCI, ERR, DQ, WMAP, with consistent units and metadata.
The output retains the first input's NFRAMES value and comment, if present; if
absent, it remains absent. The merger does not check agreement across inputs.
Inspect detector overlap: detector arrays sum before exposure averaging.
In direct Python `mask_config`, `frame_idx` is a zero-based input-frame index:
for two input frames, valid indices are 0 and 1.
An out-of-range index raises before loader cache or FITS access; an omitted or
null selector applies to all frames. YAML uses `frame` names, not `frame_idx`.
Caller setup can already have created a log or diagnostic PDF.

Review the half-open, zero-based wavelength-plane intervals `nrs1_region` and
`nrs2_region` against your cubes; both are required detector-coverage checks.
The center `[37,32]` is zero-based `[NAXIS1,NAXIS2]`, hence `SCI[:,32,37]`;
the same voxel is `[38,33]` in single-IFU YAML. Replace this illustrative center.
The merger assumes 0.05 arcsec spatial pixels. Radius selects diagnostics only;
`plot_filename: ""` disables plots; neither it nor radius crops the full cube.

Run from the checkout root after editing the copy:

```sh
cp examples/merge.yaml merge.local.yaml
# Edit merge.local.yaml; review every path, grid, region and center.
mkdir -p outputs
mkdir outputs/merge-01
hyphenate merge.local.yaml
```

Use a new directory/basename each run: outputs/logs overwrite and caches lack full
input provenance. Relative outputs resolve from the launch directory. After the
pending correction, expect `merged.fits.gz`, `merged.log`, `merged_cached_v2.npz`
in `outputs/merge-01`; no PDF or gap cache. Forced reload still writes array caches.
`mask_gap_union:false` bypasses extra union-gap masking. With union masking enabled,
an exposure with usable weights but no detected gap adds no gap to the union;
an exposure without usable weights contributes no data. Only complete, valid
gap pairs form the union. DQ/nonpositive-weight/nonfinite/zero-flux and detector-
coverage filtering still apply.
For a valid gap union, every channel from the first through the last cached gap
channel is masked in every exposure, including both endpoints. Gap indices are
zero-based; a single-channel gap at index 2 has the inclusive pair `(2, 2)`.
If no frame has a valid edge pair, the union step adds no gap mask.
New gap caches retain the same array shape and inclusive indices, using a NaN
pair for no detected gap or no usable weights; older code may not support this
sentinel. Existing valid gap pairs remain readable. Recompute with a fresh
output basename when inputs or settings change: caches do not verify provenance,
and forced gap recomputation still attempts to read an existing gap cache first.
Inspect SCI/ERR, finite masks and zero WMAP coverage; WMAP is coverage, not inverse
variance. DQ is inherited from the first cube, not recomputed. ERR excludes input
covariance and fitted-model uncertainty. A completed command is not science acceptance.
