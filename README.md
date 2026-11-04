# hyphen

Hyphen fits and removes spectral wiggles from JWST/NIRSpec data using detector
trace geometry and reference spectra, and combines corrected cubes.
Single and batch IFU processing and frame merging are the intended supported
workflows; MSA processing is experimental.

## Install from a checkout

The repository is currently private. Obtain a checkout with the necessary
access. The following HTTPS clone is a proposed release instruction; anonymous
access will depend on the repository becoming public. The planned `v1.0.0` tag
has not yet been created; before release, use an approved commit instead. This
complete installation recipe has not yet been executed against the final release
candidate.

The commands below use a POSIX shell and a `python3.13` executable with `venv`
support; use the appropriate interpreter command for your environment.

```sh
git clone https://github.com/EmilyLiepold/hyphen.git &&
cd hyphen &&
git checkout v1.0.0 &&
python3.13 -m venv .venv &&
. .venv/bin/activate &&
python -m pip install . &&
hyphenate --help
```

This requires Git, Python, pip and access to the package's build and runtime
dependencies. Package metadata specifies Python 3.12 or newer as a syntax
floor, not a tested compatibility range. An earlier dependency-complete sdist
installation smoke passed on CPython 3.13.11/macOS 26.2 arm64; it used an earlier
development snapshot plus PyYAML, not the final candidate. Other Python and
platform combinations remain unverified. Hyphen uses the [BSD 3-Clause license](LICENSE),
with copyright 2026 Emily Liepold. Citation and data redistribution terms remain
pending.

## First IFU run

This example is **unexecuted** against the final candidate. Supply complete,
matched calibrated detector CRF and rectified s3d cube FITS for the same
exposure and detector, including their WCS/ASDF, DQ and coverage extensions.
No example data are bundled. From the checkout root, copy the template and edit
the input paths, center and all data-dependent science settings before running:

```sh
cp examples/ifu_single.yaml ifu.local.yaml
# Edit ifu.local.yaml before proceeding.
mkdir -p outputs
mkdir outputs/run-01 && hyphenate ifu.local.yaml
```

Paths in the YAML resolve from the directory where the command runs. Choose a
fresh output directory and use the same path in the YAML; files and logs can be
overwritten. With the template's frame, detector and output names unchanged,
expect `outputs/run-01/00001_nrs1_hyphenated` with `.fits.gz`, `.log`, `.pdf`,
`.pkl.gz.npz` and `.pkl.gz.json` suffixes. Inspect the log, coverage and spectra;
the existence of these products does not establish fit quality or preservation
of real features. The illustrative settings need review for your target.

See the [single-IFU guide](docs/ifu-single.md), [batch-IFU guide](docs/ifu-batch.md),
[configuration reference](docs/configuration.md) and [output guide](docs/ifu-outputs.md)
for coordinates, units, chaining, diagnostics and saved-fit limitations. The
[merging guide](docs/merging.md) requires compatible registered cubes and retains
a scientific-validation caveat; review it before use.

For merger pixel masks, see [merger mask options](docs/merger-mask-options.md).

For merger cache behavior, see [merger caches](docs/merger-caches.md).
