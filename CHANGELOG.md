# Changelog

## 1.0.0 — planned, unreleased

These notes describe the selected 1.0 changes. Final release validation and
publication are still pending.

**Workflows and configuration**

- Support single and batch IFU fitting and removal of spectral wiggles, plus
  frame merging on compatible, aligned spatial and wavelength grids.
- Add configuration, IFU, output and merger guides, with YAML examples and a
  draft four-exposure rv2b benchmark recipe and recorded checksum list.
- Use consistent IFU defaults in single, batch and direct Python calls:
  weight-masking threshold 0.5 and null-model improvement threshold -0.1
  (the null-model test is disabled). Explicit null disables weight masking.
- Declare PyYAML directly and require Python >=3.12 for the source syntax.

**Fitting and saved products**

- Correct the analytic model Jacobian and key model-evaluation caches by
  array contents.
- Handle automatic-center sentinels consistently for lists, tuples and arrays,
  and account for right-ascension wrapping when selecting a trace.
- Correct retry target selection and diagnostic plotting when no fit is adopted.
  A valid all-zero residual FFT passes the Fourier check; an unavailable
  diagnostic is logged as unassessed and treated as a bad fit.
- Save initial-pass IFU fit sidecars even when plotting is disabled. Saved-fit
  replay requires the current sampled-pixel count, order and coordinates to match.

**Frame merging**

- Treat input ERR as standard deviation, combine independent-input variances,
  and write standard-deviation ERR. Exclude masked-detector variance as well
  as its SCI and weight.
- Form gap unions from complete valid pairs, including both endpoints. An
  exposure without a detected gap adds no gap; missing coverage adds no data.
- Write NaN SCI/ERR, zero WMAP and DO_NOT_USE where no contributors survive,
  retaining other first-input DQ bits there. Surviving coverage has output DQ zero.
- Preserve the first input's NFRAMES value, comment or absence.
- Reject deprecated mask_file and mask_filename YAML keys, including null,
  with guidance to mask_config. Validate frame selectors before loader I/O.
- Warn when cached arrays or gap edges are used without verified input identity.

**Limits to keep in mind**

- MSA is experimental. General resampling and further NRS2 fit tuning are outside
  this release; fitting settings and real spectral features need data-specific review.
- ERR does not include covariance or uncertainty from the fitted wiggle model.
  The merger's zero-SCI check occurs after detector summation within each frame.
- Cache warnings do not validate freshness; an existing malformed cache can fail
  before force-reload controls apply. Use fresh run-specific output paths.
- Saved fits and PDFs retain the initial pass and may differ from final SCI after
  retries. Sidecars do not authenticate inputs or the NPZ/JSON pair; load only
  trusted files. Replay still writes products and can refit when retries are enabled.
- Python >=3.12 is a syntax floor, not a tested compatibility matrix. Earlier
  dependency-complete installation evidence covers one CPython 3.13.11/macOS arm64
  stack on an earlier source; final-candidate installation remains to be checked.
- Benchmark data access, final reference products and a clean-user walkthrough
  remain to be finalized. File presence or a completion message is not science acceptance.
