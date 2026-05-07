# SOURCE PACKAGE KNOWLEDGE BASE

**Generated:** 2026-05-05
**Parent:** ../../AGENTS.md

## OVERVIEW
Core dewiggling algorithms, pipeline orchestration, and JWST data I/O for the hyphen package.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add dewiggle step | `dewiggler_stack.py` | 2500+ lines; core orchestration + all plotting |
| Fit model logic | `dewiggle_lambda_space.py` | Legendre basis, `build_wiggly_model`, `curve_fit` |
| CLI entry | `hyphenate.py` | argparse; dispatches to YAML targets |
| Frame merging | `frame_merger.py` | Co-adds, gap masks, detector union; runnable as script |
| YAML parsing | `yaml_helpers.py` | Multi-doc layering (`yaml_layerer`), step dispatch |
| Progress UI | `helpers.py` | `wigglebar` (ascii `~◊-`), `ProgressParallel` |
| Trace geometry | `tracer_helpers.py` | `find_bars`, `trace_model` |
| Step defaults | `*_step.py` (4 files) | `default_options` dict + `necessary_arguments` tuple |

## ANTI-PATTERNS (THIS DIRECTORY)
- `dewiggler_stack.py`: Deprecated module-level `_wcs_cache` to avoid repeated ASDF handle loading.
- `frame_merger.py`: `assert False` at line 200 for error handling in `determine_gap_edges`.
- `frame_merger.py`: Hardcoded pixel size `0.05` with TODO to read from headers.
- Inconsistent target pixel indexing across modules (1-indexed `[slice, pixel]` vs 0-indexed).

## NOTES
- WCS cached in module-level `_wcs_cache` to avoid repeated ASDF handle loading.
- Array caching via `np.savez_compressed` / `np.load` for expensive intermediates.
- Batch steps use `joblib.Parallel` wrapped in `ProgressParallel`.
- `dill` used instead of `pickle` for serialization compatibility.
- Output conventions: `.fits.gz` for cubes, `.pdf` for diagnostics, `.log` for runtime logs.
- See parent AGENTS.md for architecture patterns (YAML step system, dual-mode IFU/MSA, fit escalation).
