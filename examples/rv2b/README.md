# Four-exposure rv2b IFU benchmark (draft)

This recipe processes frames `00001` and `00002` for NRS1 and NRS2. It is
unexecuted as a public recipe. The data host, access, redistribution rights and
final reference are not established. The listed historical outputs are candidate
references, not accepted release oracles. Do not publish the data until rights
and hosting are confirmed.

## Inputs and setup

Use an approved Hyphen candidate and its recorded environment. Obtain the exact
four detector CRFs, four input s3d cubes, and four historical outputs listed in
`files.sha256`; their current public availability is unknown. They are locally
modified or generated products, so an archive filename alone does not establish
equivalence. Keep each FITS file complete, including WCS/ASDF and all extensions.
The 12 compressed files were previously recorded as totaling 742,516,323 bytes.
Hashes in `files.sha256` were copied from the 2026-10-22 inventory; they have
not been freshly verified for this draft.

In a fresh working directory, copy `benchmark.yaml` and `files.sha256` here,
and place the 12 exact files under the relative `data/` paths in the checksum
file. Paths resolve from the working directory, not the YAML's directory. Verify
all files before proceeding; `shasum` is available on macOS:

```sh
shasum -a 256 -c files.sha256
```

Stop if any file is missing or fails. Before a run, obtain an agreed resource
budget and record the exact Hyphen revision, dependency versions, platform and
thread settings. The template specifies `n_jobs: 6`, as in a later recorded
replay; this is not authorization to run it. The historical configuration used
10 workers. The template fixes center `[38, 33]`, fitting radius 0.5 arcsec,
central reference radius 0.2 arcsec, background radii [1.0, 1.5] arcsec, and
plotting for all targets. Other explicit settings are in the YAML. At base
`05d9e63`, the parser leaves `fft_order_width=0.1`, `power_ratio_threshold=3.0`
and `px_per_slice=2` to the fitting function defaults. Verify those defaults in
the chosen candidate. The original historical code and environment are not
fully identified.

## Run and inspect

In the working directory, make a new `generated` directory; if it already exists,
stop and choose a new working directory so that prior outputs and caches are
preserved. Then run all four cases in **one fresh process**:

```sh
mkdir generated && hyphenate benchmark.yaml
```

The parser runs `00001_nrs1`, `00001_nrs2`, `00002_nrs1`, `00002_nrs2`, in
that order. NumPy's seed 67 is set once on import, and trace construction uses
its random stream; changing the order or running a separate process for each
case can change results. Within each case the worker count is set by `n_jobs`.
Expect four `generated/*_s3d_hyphenated.fits.gz` cubes and diagnostic PDFs,
logs and fit sidecars. Check each output and the logs; a completion message or
file presence alone does not establish a successful or scientifically valid fit.

Compare each exposure against its corresponding candidate historical cube,
including SCI differences, common-finite samples, changed NaN masks, fitted
coverage, ERR/DQ/WMAP and relevant WCS metadata. Compare arrays and metadata,
not gzip bytes or timestamps. The final reference, comparison tool and scientific
acceptance tolerances remain to be chosen. The historical 16-frame merged cube
is not a reference for a four-exposure merge; this recipe does not perform a
merge. A clean-environment walkthrough and a newly budgeted replay remain needed.
