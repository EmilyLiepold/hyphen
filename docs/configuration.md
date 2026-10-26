# How YAML settings resolve

Run a reviewed recipe with `hyphenate path/to/recipe.yaml`. Relative data and
output paths resolve from the launch working directory, not the YAML directory.
The CLI has no dry-run mode; disabling fitting can still write outputs.
Single/batch IFU and frame merging are intended supported workflows; MSA remains
experimental. This page explains configuration plumbing, not science settings.

Use mapping documents separated by `---`. A `step_type: generic` document supplies
shared settings to subsequent steps until another generic document **replaces**
it completely; repeat any shared settings still needed. Earlier steps are not
changed retroactively. Each processing step document must contain its own `label`
(the dispatch print reads it directly), its registered `step_type` (`dewiggle_ifu`, `batch_dewiggle_ifu`, `merge`; MSA variants are
`dewiggle_msa`, `batch_dewiggle_msa`) and their required input/output settings.

Each option lookup starts with its fallback, then applies target defaults,
the current generic document, the step document and internal override mappings,
in that order. Later values win. An omitted key keeps the earlier value;
`key: null` supplies Python `None`, overriding it. Whether `None` is meaningful
depends on the option's consumer; it does not mean “use the default.”

Required-name checking only searches generic/step documents, excluding defaults.
A present key with `null` passes that check. Values, types, paths, collision safety
and scientific suitability are not validated by these helpers. Documents without
`step_type`, and unknown types, are silently skipped. Recognized steps missing a
required name log an error and are skipped; later documents can still run.
Malformed document shapes and errors from target functions can raise exceptions.
These are current behaviors, not an endorsed failure/recovery policy. Inspect
logs and expected products: the CLI completion message does not prove every
requested step ran.

## A contained layering example

From the source checkout root, activate a Python environment containing PyYAML
and run this exact command. It registers a custom `probe` callback, not a CLI
processing step, writes a temporary YAML file and removes it afterwards. It does
not import science orchestration, read FITS, fit spectra or create science outputs.

```sh
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from hyphen.yaml_helpers import ingest_yaml

def show(get):
    print(get("label"), get("n_jobs"),
          get("n_jobs", override=[{"n_jobs": 4}]),
          get("limit"), get("old_marker", "absent"))

recipe = """step_type: generic
n_jobs: 2
old_marker: retained
---
step_type: probe
label: first
n_jobs: 3
limit: null
---
step_type: generic
n_jobs: 5
---
step_type: probe
label: second
"""
target = {"target_step_type": "probe", "target_function": show,
          "default_options": {"n_jobs": 1, "limit": 9},
          "necessary_arguments": ["label"]}
with TemporaryDirectory() as directory:
    path = Path(directory) / "probe.yaml"
    path.write_text(recipe)
    ingest_yaml(path, targets=[target])
PY
```

Expected output:

```text
Beginning step first
first 3 4 None retained
Beginning step second
second 5 4 9 absent
```

Match every line. The result checks precedence, explicit null and generic
replacement only; it does not establish real-workflow execution, fit quality,
installation compatibility or acceptance of science thresholds or failure policy.
