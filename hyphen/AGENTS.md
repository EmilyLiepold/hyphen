# Source package map

Follow the [repository instructions](../AGENTS.md), which are authoritative.
This file is navigation; inspect the relevant code before changing its contracts.

## Where to look

| Concern | Source and entry points |
| --- | --- |
| CLI and workflow registration | [hyphenate.py](hyphenate.py): `hyphenate` |
| YAML documents and option layering | [yaml_helpers.py](yaml_helpers.py): `ingest_yaml`, `yaml_layerer`, `validate_yaml_arguments` |
| Workflow defaults and required option names | [single IFU](single_dewiggle_ifu_step.py), [batch IFU](batch_dewiggle_ifu_step.py), [single MSA](single_dewiggle_msa_step.py), [batch MSA](batch_dewiggle_msa_step.py): `*_default_options`, `*_necessary_arguments` |
| IFU/MSA orchestration, FITS/WCS, diagnostics and retries | [dewiggler_stack.py](dewiggler_stack.py): `dewiggler_stack_ifu`, `dewiggler_stack_msa`, `parse_yaml_arguments_and_*` |
| Reference spectra | [dewiggler_stack.py](dewiggler_stack.py): `determine_reference_spectrum`, `InterpolatedPf` |
| Spectral models, fitting and fit plots | [dewiggle_lambda_space.py](dewiggle_lambda_space.py): `build_wiggly_model`, `dewiggle_lambda_space`, `plot_dewiggle_result` |
| Bars and detector traces | [tracer_helpers.py](tracer_helpers.py): `find_bars`, `trace_pixels`, `trace_model`; [dewiggler_stack.py](dewiggler_stack.py): `get_2d_trace_model` for sky/WCS trace selection |
| Frame/detector masks, gaps, caching and merging | [frame_merger.py](frame_merger.py): `load_frames`, `frame_merge`, `parse_yaml_arguments_and_merge`; merge defaults are in this file |
| Progress helpers | [helpers.py](helpers.py): `wigglebar`, `ProgressParallel` |

IFU saved-fit readers and writers are `_load_fit_results` and `_save_fit_results`
in [dewiggler_stack.py](dewiggler_stack.py). They use paired compressed NPZ and
JSON files; loading enables pickle, so use only trusted artifacts.

This map does not establish workflow support or scientific validation. Check the
relevant documentation and evidence under the repository's review rules.
