"""Compare SCI spectra from one matched individual exposure; never merge cubes.

With NumPy and Astropy installed, replace /WORK and the other path placeholders
with real paths, then run these commands in order before any Python/Astropy import:
  mkdir -p /WORK/runtime/config/astropy /WORK/runtime/cache/astropy
  export XDG_CONFIG_HOME=/WORK/runtime/config XDG_CACHE_HOME=/WORK/runtime/cache
  python -B /PATH/TO/compare_spectra.py --baseline BASE.fits.gz --candidate FIXED.fits.gz \
    --case 00001_nrs1 --output-dir /WORK/comparison/00001_nrs1 --work-root /WORK
The output directory must be new and below work-root; input aliases are refused.
For each spatial pixel, use wavelengths finite in BOTH SCI arrays (excluding NaN
or infinite detector-gap samples). Report R = RMS(candidate-baseline)/median(baseline)
on that same intersection; this is not a pointwise fractional residual. R > 0.005
is flagged (strictly >0.5%); no flux/SNR cut is applied. Empty intersections and
non-positive baseline medians yield an undefined ratio. Statuses empty, zero_median
and negative_median require review; flagged=False for them is not a pass. Inspect
the finite-mask counts. SCI shapes, BUNIT, basic WCS cards and primary detector/
grating/filter identities must match (including card presence). Cards missing in
both inputs are accepted; full coordinate validity is not established thereby.

Outputs: pixels.csv (all pixels, descending defined R), flagged_pixels.csv (only
R > 0.005), and summary.json (formula, counts and input hashes). CSV coordinates
are zero-based NumPy row,column and one-based FITS x,y. RMS values use SCI units;
R is dimensionless. A small successful comparison has the expected CSV rows and
summary counts, not necessarily zero flags. Run once for each of the four cases.
Gzipped FITS arrays are loaded in memory; row-sized temporaries limit extra memory
(two float64 3814x71x83 arrays alone occupy about 360 MB). The setup above creates
and exports local Astropy config/cache paths before import; HOME is unchanged. This compares SCI only;
it does not validate full WCS/ASDF equivalence, ERR/DQ/WMAP, fits or scientific
acceptance. A zero exit means comparison completed, even when pixels are flagged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

THRESHOLD = 0.005
PRIMARY_CARDS = ["DETECTOR", "GRATING", "FILTER"]
HEADER_CARDS = ["BUNIT"] + [f"{key}{i}" for key in
    ("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CDELT") for i in range(1, 4)] + [
    f"{key}{i}_{j}" for key in ("PC", "CD") for i in range(1, 4) for j in range(1, 4)]


def spectrum_rows(baseline, candidate, case):
    if baseline.ndim != 3 or baseline.shape != candidate.shape or 0 in baseline.shape:
        raise ValueError("SCI arrays must have identical (wavelength,row,column) shapes")
    for row in range(baseline.shape[1]):
        a = np.asarray(baseline[:, row, :], dtype=np.float64)
        b = np.asarray(candidate[:, row, :], dtype=np.float64)
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        common = finite_a & finite_b
        counts = common.sum(axis=0)
        clean_a = np.where(common, a, 0)
        delta = np.where(common, b, 0) - clean_a
        with np.errstate(over="raise", invalid="raise"):
            difference = np.sqrt(np.divide((delta ** 2).sum(axis=0), counts,
                out=np.zeros(a.shape[1]), where=counts > 0))
        if not np.all(np.isfinite(difference)):
            raise ValueError("Nonfinite derived RMS; inspect input magnitudes")
        for column, count in enumerate(counts):
            median = float(np.median(a[common[:, column], column])) if count else None
            if median is not None and not np.isfinite(median):
                raise ValueError("Nonfinite baseline median; inspect input magnitudes")
            status = "empty" if count == 0 else "zero_median" if median == 0 else "negative_median" if median < 0 else "ok"
            ratio = float(difference[column] / median) if status == "ok" else None
            if ratio is not None and not np.isfinite(ratio):
                raise ValueError("Nonfinite fractional RMS; inspect input magnitudes")
            yield dict(case=case, row=row, column=column, fits_x=column + 1,
                fits_y=row + 1, common_samples=int(count),
                baseline_finite=int(finite_a[:, column].sum()),
                candidate_finite=int(finite_b[:, column].sum()),
                finite_mask_disagreements=int((finite_a[:, column] != finite_b[:, column]).sum()),
                difference_rms=float(difference[column]) if count else None,
                baseline_median=median,
                fractional_residual=ratio, flagged=ratio is not None and ratio > THRESHOLD, status=status)


def compare(baseline, candidate, case, output_dir, work_root):
    baseline, candidate = Path(baseline).resolve(strict=True), Path(candidate).resolve(strict=True)
    root, output = Path(work_root).resolve(strict=True), Path(output_dir).resolve()
    if baseline.samefile(candidate):
        raise ValueError("Baseline and candidate must not alias the same input")
    if output == root or not output.is_relative_to(root):
        raise ValueError("Output directory must be below work-root")
    if output.exists() or Path(output_dir).is_symlink():
        raise ValueError("Output directory already exists; choose a fresh path")
    with fits.open(baseline, memmap=False) as old, fits.open(candidate, memmap=False) as new:
        for extension, cards in (("SCI", HEADER_CARDS), (0, PRIMARY_CARDS)):
            before, after = old[extension].header, new[extension].header
            for key in cards:
                if (key in before) != (key in after) or before.get(key) != after.get(key):
                    raise ValueError(f"Header mismatch in {extension}: {key}")
        rows = list(spectrum_rows(old["SCI"].data, new["SCI"].data, case))
        shape, unit = list(old["SCI"].data.shape), old["SCI"].header.get("BUNIT")
    rows.sort(key=lambda r: (r["fractional_residual"] is None, -(r["fractional_residual"] or 0), r["row"], r["column"]))
    inputs = []
    for role, path in (("baseline", baseline), ("candidate", candidate)):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        inputs.append(dict(role=role, path=str(path), bytes=path.stat().st_size, sha256=digest))
    summary = dict(case=case, shape=shape, sci_unit=unit, inputs=inputs,
        formula="RMS(candidate-baseline) / median(baseline), common finite SCI samples",
        threshold=THRESHOLD, comparison="strictly greater than",
        checked_header_cards=dict(SCI=HEADER_CARDS, PRIMARY=PRIMARY_CARDS),
        pixels=len(rows), flagged_pixels=sum(r["flagged"] for r in rows),
        compared_pixels=sum(r["status"] == "ok" for r in rows),
        empty_pixels=sum(r["status"] == "empty" for r in rows),
        zero_median_pixels=sum(r["status"] == "zero_median" for r in rows),
        negative_median_pixels=sum(r["status"] == "negative_median" for r in rows),
        zero_median_nonzero_difference_pixels=sum(r["status"] == "zero_median" and
            r["difference_rms"] > 0 for r in rows),
        finite_mask_changed_pixels=sum(r["finite_mask_disagreements"] > 0 for r in rows),
        finite_mask_disagreements=sum(r["finite_mask_disagreements"] for r in rows))
    output.mkdir(parents=True)  # Atomic refusal if another process created the run directory.
    for filename, selected in (("pixels.csv", rows), ("flagged_pixels.csv", [r for r in rows if r["flagged"]])):
        with (output / filename).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    with (output / "summary.json").open("x") as handle:
        handle.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("baseline", "candidate", "output-dir", "work-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.baseline, args.candidate, args.case, args.output_dir, args.work_root), indent=2))
    except (OSError, ValueError, KeyError, FloatingPointError) as error:
        parser.exit(1, f"Comparison refused: {error}\n")


if __name__ == "__main__":
    main()
