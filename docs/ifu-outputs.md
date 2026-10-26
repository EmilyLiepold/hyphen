# Read individual IFU outputs

Use this after the [single](ifu-single.md) or [batch](ifu-batch.md) recipe.
Single/batch IFU and frame merging are intended supported workflows; MSA is
experimental. This page covers individual exposures, before merging.

## Cube and log

The output keeps the input cube's HDUs and units. With an adopted fit, SCI is
the input spectrum minus the wiggle model, with the continuum retained.
Reference construction first sets SCI samples with nonzero DQ to NaN across
the cube, including outside the correction region. Other unselected SCI stays
unchanged. Additional fitting masks select fit samples; they do not promise
those samples become NaN in the output.

With `mask_bad_fits: true`, a fit still rejected after retry has its entire SCI
spectrum set to NaN. `DO_NOT_USE` is added to every DQ sample in that spectrum,
preserving existing bits. The same applies when retry reports no finite input
SCI, no samples after masking with valid normalization, or no positive weights
when weight masking is enabled. Processing continues with the next spectrum.
A successful retry retains its corrected SCI and adds no DQ bits.

Other errors escaping the fitter's existing handlers propagate and prevent the
final cube write; invalid reference normalization is one example. Initial-pass
errors still propagate. Missing FFT diagnostics and no-adopted-fit results need
separate handling; this retry correction does not resolve them. With
`mask_bad_fits: false`, no retry or new failure DQ flags are added. ERR and WMAP
are retained, so ERR excludes uncertainty from the fitted wiggle correction.

Read the `.log` alongside spectra: it records spaxel coordinates, radius in
arcseconds, trial/adopted orders, fit statistics and quality/retry messages.
A completion message alone does not establish corrected coverage or fit quality.

## Diagnostic PDF

With plotting enabled, inspect reference-spectrum and detector-bar pages, then
each processed spaxel's position/trace map and spectral pages. The overview
median image is generated before PDF assembly; older assembly paths can omit it
from the final PDF. Check expected pages rather than inferring success from their absence.

The wavelength panel shows input, full model, wiggle model and corrected
spectrum in input flux units. The detector-coordinate panel divides by the
central reference model and then by the median of that ratio. It shows the
normalized input/model, central/background components and corrected spectrum.
The residual panel compares continuum-subtracted input and corrected spectra
in those normalized units. The detector-coordinate and residual panels use
fit-selected samples: finite normalized flux, DQ zero, strict 1st–99th percentile cuts, plus configured
weight and wavelength masks. Excluded wavelengths can still appear in SCI.

The title's “RMS residual” is `100 * sqrt(ν)` displayed as a percent, where
`ν = sum(((model - normalized_flux) / normalized_noise)**2) / (N - K)`;
N is the selected sample count and K the fitted parameter count. The IFU path
uses unit normalized noise, not the cube's ERR. Thus √ν is a degrees-of-freedom
adjusted normalized residual measure, not an ERR-weighted uncertainty test.
A separate comparison of two fitted cubes,
`RMS(candidate-reference) / median(reference)` over jointly finite wavelengths,
measures a spectral change. Residual guides are at ±√ν: a historical “5% residual”
legend does not mean their height is fixed at 0.05.

The Fourier panel shows magnitudes, not power spectral density: both input and
residual transforms are divided by the maximum input-transform magnitude among
retained frequencies (absolute frequency <5). They follow continuum subtraction
and interpolation onto a fine detector grid;
frequency is cycles per detector pixel. A low residual peak alone does not prove
preservation of real spectral features; empty/zero transforms need separate review.

## Saved fits and replay

For `name.fits.gz`, expect `name.pkl.gz.npz` and `name.pkl.gz.json` when plots
are enabled. Current saving depends on plotting. Keep the pair, original inputs,
resolved settings, environment and revision together. JSON rows and NPZ rows
must stay aligned: never independently sort or substitute one file. Loading
allows pickle; use only trusted artifacts. Do not treat this pair as a complete
provenance record or assume a legacy DILL file is interchangeable.

Saved fits and PDFs capture the initial pass, before the later quality/retry
loop; they can differ from final SCI after retries. `perform_dewiggling: false`
loads saved fits but still writes a cube and can refit if `mask_bad_fits: true`.
Replay needs matching input data, selected spaxel order and settings, including
`diagnostic_fraction`; it is not a dry run or a guaranteed no-refitting mode.

## Inspect one spectrum

This read-only example is unexecuted against science data. In an environment
with Astropy, NumPy and JWST's `stdatamodels`, replace both placeholder paths and
zero-based row/column arguments. This inspects only `SCI[:,32,37]` and its ancillary
arrays, not the whole cube. DQ must use unsigned 32-bit integers (either byte
order); other dtypes are rejected. Added flags are allowed only for `DO_NOT_USE`
on NaN output samples; existing flags must not be cleared.

```sh
python - INPUT_CUBE.fits.gz OUTPUT_CUBE.fits.gz 32 37 <<'PY'
import sys
import numpy as np
from astropy.io import fits
from stdatamodels.jwst.datamodels import dqflags
r, c = map(int, sys.argv[3:])
with fits.open(sys.argv[1]) as a, fits.open(sys.argv[2]) as b:
    shape = a["SCI"].shape
    if len(shape) != 3 or shape != b["SCI"].shape or not (0 <= r < shape[1] and 0 <= c < shape[2]):
        raise ValueError("SCI shape mismatch or invalid row/column")
    x, y = (h["SCI"].section[:, r, c] for h in (a, b))
    for ext in ("ERR", "WMAP"):
        if a[ext].shape != b[ext].shape or not np.array_equal(
                a[ext].section[:, r, c], b[ext].section[:, r, c], equal_nan=True):
            raise ValueError(f"Unexpected {ext} change in inspected spectrum")
    if a["DQ"].shape != shape or b["DQ"].shape != shape:
        raise ValueError("DQ shape mismatch")
    old, new = (h["DQ"].section[:, r:r+1, c:c+1][:, 0, 0] for h in (a, b))
    if any(z.dtype.kind != "u" or z.dtype.itemsize != 4 for z in (old, new)):
        raise ValueError("DQ must be unsigned 32-bit integers")
    old, new = (np.asarray(z, dtype=np.uint32) for z in (old, new))
    added, cleared = new & ~old, old & ~new
    print("DQ added/cleared samples:", np.count_nonzero(added), np.count_nonzero(cleared))
    print("DQ added/cleared bit masks:", np.unique(added), np.unique(cleared))
    if np.any(cleared) or np.any(added & ~np.uint32(dqflags.pixel["DO_NOT_USE"])) or np.any((added != 0) & ~np.isnan(y)):
        raise ValueError("Unexpected DQ change in inspected spectrum")
    print("SCI shape:", shape, "units:", a["SCI"].header.get("BUNIT"), b["SCI"].header.get("BUNIT"))
    print("Finite input/output:", np.isfinite(x).sum(), np.isfinite(y).sum())
    print("Changed finite mask:", np.count_nonzero(np.isfinite(x) != np.isfinite(y)))
PY
```

Expect matching shape/units and unchanged sampled ERR/WMAP. A failed masked
spectrum should be all NaN with the added flag and no cleared bits. A recovered
fit should add no flags. Review the printed masks and log; other NaNs can arise
from input DQ. This check permits documented flag additions but does not prove
that every failed spectrum was flagged. It does not validate WCS, other spaxels,
fit quality, spectral-feature preservation or propagated uncertainty.
