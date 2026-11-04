# hyphen

Hyphen fits and removes spectral wiggles from JWST/NIRSpec data using detector
trace geometry and reference spectra, and combines corrected cubes.
Single and batch IFU processing and frame merging are the intended supported
workflows; MSA processing is experimental.

## Install from a checkout

Obtain a checkout with the necessary access. The checkout command below requires
the `v1.0.0` tag; if it is unavailable, replace it with an approved commit. This
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
floor, not a tested compatibility range. An earlier 1.0 candidate wheel and
source distribution were installed with dependencies and smoke-checked on CPython
3.13.11/macOS 26.2 arm64. These checks describe one observed environment, not a
tested compatibility range. Hyphen uses the [BSD 3-Clause license](LICENSE),
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
expect files with the prefix `outputs/run-01/00001_nrs1_hyphenated` and suffixes
`.fits.gz`, `.log`, `.pdf`, `.pkl.gz.npz` and `.pkl.gz.json`. Inspect the log,
coverage and spectra;
the existence of these products does not establish fit quality or preservation
of real features. The illustrative settings need review for your target.

See the [single-IFU guide](docs/ifu-single.md), [batch-IFU guide](docs/ifu-batch.md),
[configuration reference](docs/configuration.md) and [output guide](docs/ifu-outputs.md)
for coordinates, units, chaining, diagnostics and saved-fit limitations. The
[merging guide](docs/merging.md) requires compatible registered cubes and retains
a scientific-validation caveat; review it before use.

For merger pixel masks, see [merger mask options](docs/merger-mask-options.md).

For merger cache behavior, see [merger caches](docs/merger-caches.md).
