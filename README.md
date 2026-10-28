# hyphen

Hyphen fits and removes spectral wiggles from JWST/NIRSpec data using detector
trace geometry and reference spectra, and combines corrected cubes.

Single/batch IFU processing and frame merging are the intended supported workflows;
MSA processing is experimental.

For an installed checkout, start with the [single-IFU launch guide](docs/ifu-single.md)
and its annotated template. The example requires your own matched input products
and remains unexecuted; its settings require scientific review for your target.

See [how YAML settings resolve](docs/configuration.md) before editing a recipe.

Read [individual IFU outputs and diagnostics](docs/ifu-outputs.md).
