# Four-exposure rv2b IFU benchmark (draft)

This Hyphen IFU benchmark processes frames `00001` and `00002` for NRS1 and
NRS2. This public recipe has not yet been walked through. Contact the authors
for data access and sharing conditions. The four corrected outputs from the
release-preparation run are the selected 1.0 regression references. Agreement
with them measures regression; assess fit diagnostics for your scientific use.

## Inputs and setup

Use Hyphen in a recorded environment. Obtain the exact
four detector CRFs, four input s3d cubes, and four selected reference cubes
listed in `files.sha256`; their current public availability is unknown. They
are locally modified or generated products, so an archive filename alone does not establish
equivalence. Keep each FITS file complete, including WCS/ASDF and all extensions.
The 12 compressed files were previously recorded as totaling 742,518,722 bytes.
Input hashes were copied from the 2026-10-22 inventory; reference hashes were
recorded after the 2026-10-27 release run and checked again after the later
warm-merger run. Verify the exact staged files before use. Reference cubes were
generated with source snapshot SHA-256
`ae2cc4b98b12a5987b1eb540fb2bd08ace4651aed1cf3a4661435e5adbfe879a`
(a source-content identity, not a Git commit), CPython 3.13.11 on macOS arm64,
six workers, and numerical thread settings of one. The source used version
0.0.1 before the planned 1.0.0 metadata update.

In a fresh working directory, copy `benchmark.yaml` and `files.sha256` here,
and place the 12 exact files under the relative `data/` paths in the checksum
file. Paths resolve from the working directory, not the YAML's directory. Verify
all files before proceeding; `shasum` is available on macOS:

```sh
shasum -a 256 -c files.sha256
```

Stop if any file is missing or fails. Record the exact Hyphen revision,
dependency versions, platform and thread settings. The template specifies
`n_jobs: 6`, as in the recorded reference run. The historical configuration
used 10 workers. The template fixes center `[38, 33]`, fitting radius 0.5 arcsec,
central reference radius 0.2 arcsec, background radii [1.0, 1.5] arcsec, and
plotting for all targets. Other explicit settings are in the YAML. At base
`05d9e63`, the parser leaves `fft_order_width=0.1`, `power_ratio_threshold=3.0`
and `px_per_slice=2` to the fitting function defaults. Verify those defaults in
the chosen candidate. The original historical code and environment are not
fully identified.

## Run and inspect

In the working directory, make a new `generated` directory; if it already exists,
stop and choose a new working directory so that prior outputs and caches are
preserved. Set the five numerical thread controls to one as shown, then run
all four cases in **one fresh process**:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir generated && hyphenate benchmark.yaml
```

The parser runs `00001_nrs1`, `00001_nrs2`, `00002_nrs1`, `00002_nrs2`, in
that order. NumPy's seed 67 is set once on import, and trace construction uses
its random stream; changing the order or running a separate process for each
case can change results. Within each case the worker count is set by `n_jobs`.
Expect four `generated/*_s3d_hyphenated.fits.gz` cubes and diagnostic PDFs,
logs and fit sidecars. Check each output and the logs; a completion message or
file presence alone does not establish a successful or scientifically valid fit.

Compare each exposure against its corresponding selected reference cube,
including SCI differences, common-finite samples, changed NaN masks, fitted
coverage, ERR/DQ/WMAP and relevant WCS metadata. Compare arrays and metadata,
not gzip bytes or timestamps. The [SCI comparison tool](../../tools/compare_spectra.py)
reports a fractional residual for each pixel; its module documentation gives the
command, output files and limits. Its 0.5% flag prompts review and is not a
scientific pass criterion. Inspect flagged spectra and assess the results for
your data. The historical 16-frame merged cube is not a reference for a
four-exposure merge; this recipe does not perform a merge. A clean-environment
walkthrough remains needed. Preserve the historical paper-run outputs and
evidence separately from these regression references.
