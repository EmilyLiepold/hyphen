# Merger caches

The frame merger saves arrays and, when union-gap masking is enabled, gap edges
using names derived from the output filename. Reusing an output name can reuse
these caches. Hyphen cannot reliably verify that a cache matches the current input
cubes or settings; a changed input or configuration can therefore yield stale
results. The merger logs a warning when it actually uses cached arrays or gaps.
For a YAML merger step, look in the output-derived `.log` file for these warnings.

To recompute after changing inputs or settings, use a fresh output filename in a
new directory. Alternatively, set both controls in the merger YAML:

```yaml
force_reload_frames: true
force_reload_gaps: true
```

These settings reload frame arrays and recompute gap edges independently; the
latter matters when union-gap masking is enabled. They still write caches under the
output-derived name. A pixel `mask_config` also forces a frame reload, but does not
force gap recomputation. Python callers use `load_frames_from_disk=True` and
`recompute_gaps=True` for the corresponding controls.

Both cache readers currently try to read an existing cache before applying the
force controls. A malformed existing cache can therefore raise an error even with
these controls set. In that case use a fresh output filename in a new directory.
The warning is not an automatic validity check, and no guarantee is made that an
existing cache matches the current inputs or settings.
