# Merger pixel-mask settings

For a merger YAML step, remove the deprecated `mask_filename` and `mask_file`
keys from both the generic and merge documents. Their presence is an error even
when their value is `null`; a merge-step `null` does not remove a generic key.
Hyphen does not convert the contents of a legacy mask file automatically.

To mask pixels, manually specify `mask_config` in the merge document. For example,
with `target_frames: [f1, f2]` and `target_detectors: [nrs1, nrs2]`:

```yaml
mask_config:
  - frame: f2
    detector: nrs1
    pixels: [[2, 1]]
```

This masks the pixel at zero-based `[NAXIS1, NAXIS2] = [2, 1]` at every
wavelength for `f2` and `nrs1`. Omit `frame` or `detector` to apply an entry to
all selected frames or detectors. Names must match the selected frame and detector
lists. Omit `mask_config` or set it to `null` when no pixel mask is needed. This
fragment is not a complete runnable configuration: a merger also needs its normal
input, output, frame and detector settings and trusted, compatible FITS cubes.

The check runs when the merger step is dispatched. YAML reading and earlier steps
in the same configuration can already have occurred. It prevents this merger step
from opening its output log or processing cubes when a legacy key is present.
