"""Tiny IFU cubes test failure SCI/DQ and real FITS writes, not fit quality."""
import hashlib
from types import SimpleNamespace

from astropy.io import fits
import numpy as np
import pytest

from hyphen import dewiggle_lambda_space as fitting
from hyphen import dewiggler_stack as stack


@pytest.mark.parametrize("outcome", ["good", "recovered", "exhausted", "unusable",
                                     "unexpected", "initial_unusable", "unmasked",
                                     "zero_weight", "invalid_reference"])
def test_ifu_failure_output_and_continuation(monkeypatch, tmp_path, outcome):
    original = np.arange(16., dtype=float).reshape(4, 2, 2) + 10
    dq = np.zeros(original.shape, dtype=np.uint32)
    dq[:, 0, 0] = [2**31, 2, 0, 8]
    source, output = tmp_path / "input.fits", tmp_path / "output.fits"
    hdus = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({"ORIGIN": "hyphen-test"})),
        fits.ImageHDU(original, name="SCI"),
        fits.ImageHDU(np.full_like(original, 2), name="ERR"),
        fits.ImageHDU(dq, name="DQ"),
        fits.ImageHDU(np.ones_like(original), name="WMAP"),
        fits.ImageHDU(np.arange(7, dtype=np.uint8), name="ASDF"),
    ])
    hdus["SCI"].header["CTYPE3"] = "WAVE"
    if outcome == "zero_weight":
        hdus["WMAP"].data[:, 0, 0] = 0
    hdus.writeto(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    opened, calls = [], []

    def load_frames(*args):
        cube = fits.open(source)
        opened.append(cube)
        return [SimpleNamespace(data=np.ones((2, 2)))] * 8, cube, "detector", "cube"

    details = dict(alpha_order=0, beta_order=0, gamma_order=0, order=0)
    fft = dict(freq=np.array([1., 2.]), residual_fft=np.ones(2))
    def fit(*args, **kwargs):
        spectrum = args[3].copy()
        target = next(t for t in [(0, 0), (1, 1)]
                      if np.array_equal(spectrum, original[:, t[0], t[1]]))
        calls.append(target)
        attempt = calls.count(target)
        if target == (0, 0):
            if outcome == "initial_unusable" or (outcome == "unusable" and attempt == 2):
                return fitting.dewiggle_lambda_space(
                    [[lambda x: x]], np.array([1.]), np.arange(4.), spectrum,
                    np.ones(4), background_pf=np.array([1.]))
            if outcome == "invalid_reference" and attempt == 2:
                return fitting.dewiggle_lambda_space(
                    [[lambda x: x]], np.array([0.]), np.arange(4.), spectrum,
                    np.zeros(4), background_pf=np.array([1.]))
            if outcome == "unexpected" and attempt == 2:
                raise KeyError("fixture programming error")
        bad = target == (0, 0) and outcome != "good" and (
            attempt == 1 or outcome == "exhausted")
        shift = 3 if attempt == 2 else 1
        return (np.ones(4), spectrum - shift, .04 if bad else .0001,
                np.arange(6.), details, fft)

    replacements = dict(
        load_frames=load_frames, determine_detector_name=lambda *a: "NRS1",
        get_wcs_ifu=lambda *a: (None, SimpleNamespace(pixel_to_world_values=lambda *a: a), None, None),
        get_2d_trace_model=lambda *a: lambda *b: (np.ones(4), np.ones(4)),
        get_principal_axes_ifu=lambda *a: (np.arange(2), np.arange(2), np.arange(4)),
        determine_spatial_arr=lambda *a: None, determine_image=lambda *a: np.ones((2, 2)),
        determine_target_pixel_ifu=lambda **k: [1, 1],
        determine_spaxel_distances=lambda *a, **k: np.zeros((2, 2)),
        determine_reference_spectrum=lambda *a, **k: (np.ones(4), None, np.ones(4)),
        find_bars=lambda *a: list(range(30)), infer_fractional_position=lambda *a: (0.1, 0.1),
        dewiggle_lambda_space=fit,
    )
    if outcome == "zero_weight":
        replacements["_load_fit_results"] = lambda *a: [
            (np.arange(6.), details, list(t), original[:, t[0], t[1]] - 1,
             fft, .04 if t == (0, 0) else .0001, "", 0) for t in [(0, 0), (1, 1)]]
    for name, value in replacements.items():
        monkeypatch.setattr(stack, name, value)
    for name in ["build_image_map_ifu", "build_reference_spectra_plot",
                 "build_bar_map_plot", "build_spaxel_map_plot_ifu"]:
        monkeypatch.setattr(stack, name, lambda *a, **k: None)
    options = dict(plot=False, n_jobs=1, use_cached_wcs=False, diagnostic_fraction=1,
                   target_wiggle_pixels=[[1, 1], [2, 2]], slices_to_use=[0],
                   null_fit_improvement_threshold=-1, fractional_residual_threshold=.1,
                   power_ratio_threshold=2, mask_bad_fits=outcome != "unmasked",
                   perform_dewiggling=outcome != "zero_weight")
    try:
        if outcome in ["unexpected", "initial_unusable", "invalid_reference"]:
            with pytest.raises(KeyError if outcome == "unexpected" else ValueError):
                stack.dewiggler_stack_ifu("detector", str(source), str(output), **options)
            assert not output.exists()
            return
        stack.dewiggler_stack_ifu("detector", str(source), str(output), **options)
        failed = outcome in ["exhausted", "unusable", "zero_weight"]
        expected = original.copy()
        expected[:, 0, 0] = np.nan if failed else original[:, 0, 0] - (
            3 if outcome == "recovered" else 1)
        expected[:, 1, 1] -= 1
        expected_dq = dq.copy()
        if failed:
            expected_dq[:, 0, 0] |= np.uint32(1)  # JWST DO_NOT_USE; retain every input bit.
        with fits.open(source) as first, fits.open(output) as last:
            np.testing.assert_array_equal(last["SCI"].data, expected)
            np.testing.assert_array_equal(last["DQ"].data, expected_dq)
            assert last["DQ"].data.dtype == np.dtype("uint32")
            assert [h.name for h in last] == [h.name for h in first]
            for left, right in zip(first, last):
                assert left.header == right.header
                if left.name not in ["SCI", "DQ"] and left.data is not None:
                    np.testing.assert_array_equal(left.data, right.data)
        expected_calls = [] if outcome == "zero_weight" else [(0, 0), (1, 1)]
        if outcome in ["recovered", "exhausted", "unusable"]:
            expected_calls.append((0, 0))
        assert calls == expected_calls
    finally:
        for cube in opened:
            cube.close()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("condition", ["nonfinite", "normalization", "dq", "mask"])
def test_numerical_fitter_identifies_unusable_samples(condition):
    lam, spectrum = np.arange(8.), np.arange(8.) + 10
    dq, mask, continuum = np.zeros(8), np.ones(8, dtype=bool), np.array([1.])
    if condition == "nonfinite":
        spectrum[:] = np.nan
    elif condition == "normalization":
        continuum[:] = 0
    elif condition == "dq":
        dq[:] = 1
    else:
        mask[:] = False
    with pytest.raises(ValueError) as error:
        fitting.dewiggle_lambda_space(
            [[lambda x: x]], continuum, lam, spectrum, dq,
            target_maskarray=mask, background_pf=np.array([1.]))
    assert type(error.value).__name__ == (
        "ValueError" if condition == "normalization" else "UnusableSpectrumError")
