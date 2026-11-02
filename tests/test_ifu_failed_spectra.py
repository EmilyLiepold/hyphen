"""Tiny IFU cubes test failure SCI/DQ and real FITS writes, not fit quality."""
import hashlib
import logging
from types import SimpleNamespace

from astropy.io import fits
import numpy as np
import pytest

from hyphen import dewiggle_lambda_space as fitting
from hyphen import dewiggler_stack as stack


@pytest.mark.parametrize("outcome", ["good", "recovered", "exhausted", "unusable",
                                     "unexpected", "initial_unusable", "unmasked",
                                     "zero_weight", "invalid_reference",
                                     "fft_missing_first", "fft_missing_later",
                                     "fft_empty_first", "fft_empty_later",
                                     "fft_zero_good", "fft_zero_bad", "fft_zero_negative",
                                     "fft_ratio_recovered", "fft_nearest",
                                     "fft_recovered", "fft_exhausted",
                                     "fft_unusable", "fft_bad_shape",
                                     "fft_plot_missing_unmasked", "fft_plot_missing_masked"])
def test_ifu_failure_output_and_continuation(monkeypatch, tmp_path, caplog, outcome):
    plotted = outcome.startswith("fft_plot_")
    nlam = 12 if plotted else 4
    lam = np.linspace(1., 2., nlam) if plotted else np.arange(nlam)
    original = np.arange(nlam * 4., dtype=float).reshape(nlam, 2, 2) + 10
    dq = np.zeros(original.shape, dtype=np.uint32)
    dq[:4, 0, 0] = [2**31, 2, 0, 8]
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
    caplog.set_level(logging.INFO, logger=stack.logger.name)
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
        if outcome in ("fft_recovered", "fft_exhausted", "fft_unusable") and attempt > 2:
            attempt = 2  # Replay retries the original saved result once.
        if target == (0, 0):
            if outcome == "initial_unusable" or (outcome in ("unusable", "fft_unusable") and attempt == 2):
                return fitting.dewiggle_lambda_space(
                    [[lambda x: x]], np.array([1.]), np.arange(4.), spectrum,
                    np.ones(4), background_pf=np.array([1.]))
            if outcome == "invalid_reference" and attempt == 2:
                return fitting.dewiggle_lambda_space(
                    [[lambda x: x]], np.array([0.]), np.arange(4.), spectrum,
                    np.zeros(4), background_pf=np.array([1.]))
            if outcome == "unexpected" and attempt == 2:
                raise KeyError("fixture programming error")
        fft_cases = outcome.startswith("fft_")
        bad = target == (0, 0) and (
            (not fft_cases and outcome != "good" and (attempt == 1 or outcome == "exhausted"))
            or (outcome == "fft_zero_bad" and attempt == 1)
        )
        result_fft = fft
        missing_first = outcome in ("fft_missing_first", "fft_recovered", "fft_exhausted", "fft_unusable",
                                    "fft_plot_missing_unmasked", "fft_plot_missing_masked")
        if (target == (0, 0) and missing_first and
                (attempt == 1 or outcome == "fft_exhausted")) or (
                target == (1, 1) and outcome == "fft_missing_later"):
            result_fft = {}
        elif ((target == (0, 0) and outcome == "fft_empty_first") or
              (target == (1, 1) and outcome == "fft_empty_later")):
            result_fft = dict(freq=np.array([]), residual_fft=np.array([]))
        elif (target == (0, 0) and outcome in ("fft_zero_good", "fft_zero_bad")) or outcome == "fft_zero_negative":
            result_fft = dict(freq=np.array([1., 2.]), residual_fft=np.zeros(2))
        elif target == (0, 0) and outcome == "fft_ratio_recovered" and attempt == 1:
            result_fft = dict(freq=np.array([1., 2.]), residual_fft=np.array([10., 1.]))
        elif target == (0, 0) and outcome == "fft_nearest":
            result_fft = dict(freq=np.array([3., 2.]), residual_fft=np.array([1., 3.]))
        elif target == (0, 0) and outcome == "fft_bad_shape":
            result_fft = dict(freq=np.array([1., 2.]), residual_fft=np.ones(3))
        shift = 3 if attempt == 2 else 1
        xsol = np.array([1., 0., 1., 0., 0., 1.]) if plotted else np.arange(6.)
        return (np.ones(nlam), spectrum - shift, .04 if bad else .0001,
                xsol, details, result_fft)

    replacements = dict(
        load_frames=load_frames, determine_detector_name=lambda *a: "NRS1",
        get_wcs_ifu=lambda *a: (None, SimpleNamespace(pixel_to_world_values=lambda *a: a), None, None),
        get_2d_trace_model=lambda *a: lambda *b: (
            (lambda x: 40 * x, lambda x: x) if plotted else (np.ones(4), np.ones(4))),
        get_principal_axes_ifu=lambda *a: (np.arange(2), np.arange(2), lam),
        determine_spatial_arr=lambda *a: None, determine_image=lambda *a: np.ones((2, 2)),
        determine_target_pixel_ifu=lambda **k: [1, 1],
        determine_spaxel_distances=lambda *a, **k: np.zeros((2, 2)),
        determine_reference_spectrum=lambda *a, **k: (
            np.ones(nlam), np.array([1.]) if plotted else None, np.ones(nlam)),
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
    if plotted:
        # Only overview/map pages are stubs; spectral and FFT plots are real.
        from matplotlib import pyplot as plt
        from pypdf import PdfReader

        def page(pdf, *args, **kwargs):
            if pdf:
                fig = plt.figure(figsize=(1, 1))
                pdf.savefig(fig)
                plt.close(fig)
        monkeypatch.setattr(stack, "build_image_map_ifu", page)
        monkeypatch.setattr(stack, "build_reference_spectra_plot", page)
    options = dict(plot=str(tmp_path / "diagnostics.pdf") if plotted else False,
                   n_jobs=1, use_cached_wcs=False, diagnostic_fraction=1,
                   target_wiggle_pixels=[[1, 1], [2, 2]], slices_to_use=[0],
                   null_fit_improvement_threshold=-1, fractional_residual_threshold=.1,
                   power_ratio_threshold=(-1 if outcome == "fft_zero_negative" else
                                          1.5 if outcome == "fft_ratio_recovered" else 2),
                   mask_bad_fits=outcome not in (
                       "unmasked", "fft_missing_first", "fft_missing_later",
                       "fft_empty_first", "fft_empty_later", "fft_plot_missing_unmasked"),
                   perform_dewiggling=outcome != "zero_weight")
    try:
        if outcome in ["unexpected", "initial_unusable", "invalid_reference", "fft_bad_shape"]:
            with pytest.raises(KeyError if outcome == "unexpected" else ValueError):
                stack.dewiggler_stack_ifu("detector", str(source), str(output), **options)
            assert not output.exists()
            return
        stack.dewiggler_stack_ifu("detector", str(source), str(output), **options)
        failed = outcome in ["exhausted", "unusable", "zero_weight",
                             "fft_exhausted", "fft_unusable"]
        expected = original.copy()
        expected[:, 0, 0] = np.nan if failed else original[:, 0, 0] - (
            3 if outcome in ("recovered", "fft_recovered", "fft_zero_bad", "fft_ratio_recovered",
                             "fft_plot_missing_masked") else 1)
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
        if outcome in ["recovered", "exhausted", "unusable", "fft_recovered",
                       "fft_exhausted", "fft_unusable", "fft_zero_bad", "fft_ratio_recovered",
                       "fft_plot_missing_masked"]:
            expected_calls.append((0, 0))
        assert calls == expected_calls
        if plotted:
            texts = [page.extract_text() for page in PdfReader(tmp_path / "diagnostics.pdf").pages]
            assert any("Fourier spectrum" in text for text in texts)
            assert any("unassessed" in record.getMessage().lower() and "(0, 0)" in record.getMessage()
                       for record in caplog.records)
        if outcome == "fft_nearest":
            assert any("P(1) / P(3/2) = 1.5000" in record.getMessage()
                       for record in caplog.records)

        replay_cases = ("fft_missing_first", "fft_missing_later", "fft_empty_first",
                        "fft_empty_later", "fft_recovered", "fft_exhausted", "fft_unusable")
        if outcome in replay_cases:
            target = (1, 1) if outcome.endswith("later") else (0, 0)
            assert any("unassessed" in record.getMessage().lower() and
                       f"({target[0]}, {target[1]})" in record.getMessage()
                       for record in caplog.records)
            # Replay the real initial-fit sidecars: missing keys are saved as empty arrays.
            base = str(output).replace(".fits", ".pkl")
            pair = [tmp_path / ("output.pkl" + suffix) for suffix in (".npz", ".json")]
            pair_before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in pair]
            rows = stack._load_fit_results(base)
            saved = rows[0 if target == (0, 0) else 1]
            assert saved[4]["freq"].size == 0 and saved[4]["residual_fft"].size == 0
            np.testing.assert_array_equal(saved[3], original[:, target[0], target[1]] - 1)
            caplog.clear()
            stack.dewiggler_stack_ifu("detector", str(source), str(output),
                                     **{**options, "perform_dewiggling": False})
            replay_retry = outcome in ("fft_recovered", "fft_exhausted", "fft_unusable")
            assert calls == expected_calls + ([(0, 0)] if replay_retry else [])
            assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in pair] == pair_before
            with fits.open(output) as replay:
                np.testing.assert_array_equal(replay["SCI"].data, expected)
                np.testing.assert_array_equal(replay["DQ"].data, expected_dq)
            assert any("unassessed" in record.getMessage().lower() and
                       f"({target[0]}, {target[1]})" in record.getMessage()
                       for record in caplog.records)
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
