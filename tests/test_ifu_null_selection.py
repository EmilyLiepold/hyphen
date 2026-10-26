"""Check IFU trace adoption against its frequency-shifted null comparison."""
from inspect import signature
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hyphen import dewiggler_stack as stack


@pytest.mark.parametrize("second_null,threshold,selected_offset", [
    (0.5, 0.125, 2),       # Better than null by more than the threshold.
    (0.375, 0.125, 1),     # Equality is insufficient: comparison is strict.
    (0.3125, 0.125, 1),    # Lower ordinary score alone is insufficient.
    (None, -1, 2),         # Negative threshold disables the null fits.
])
def test_ifu_null_comparison_selects_written_spectrum(
    monkeypatch, tmp_path, second_null, threshold, selected_offset
):
    original = np.arange(12., dtype=float).reshape(3, 2, 2) + 10
    written = []

    class Cube(list):
        def writeto(self, *args, **kwargs):
            written.append(self[1].data.copy())

    def load_frames(*args):
        cube = Cube([None, SimpleNamespace(data=original.copy()), None,
                     SimpleNamespace(data=np.zeros_like(original)),
                     SimpleNamespace(data=np.ones_like(original))])
        return [SimpleNamespace(data=np.ones((2, 2)))] * 8, cube, "detector", "cube"

    details = dict(alpha_order=0, beta_order=0, gamma_order=0, order=0)
    fft = dict(freq=np.array([1.]), spectrum_fft=np.array([2.]), residual_fft=np.array([1.]))
    # Scores below are sqrt(nu), not nu; every square is exact in binary.
    outcomes = [(0.5, 1), (0.25, 2)] if threshold < 0 else [
        (0.75, 100), (0.5, 1), (second_null, 200), (0.25, 2)]
    remaining = iter(outcomes)

    def fit(*args, **kwargs):
        score, offset = next(remaining)
        return np.ones(3), args[3] - offset, score**2, np.arange(6.), details, fft

    assert signature(stack.dewiggle_lambda_space).parameters["frequency_fudge"].default == 1
    fitter = Mock(side_effect=fit)
    replacements = dict(
        load_frames=load_frames,
        determine_detector_name=lambda *a: "NRS1",
        get_wcs_ifu=lambda *a: (None, SimpleNamespace(pixel_to_world_values=lambda *a: a), None, None),
        get_2d_trace_model=lambda *a: lambda *b: (np.ones(3), np.ones(3)),
        get_principal_axes_ifu=lambda *a: (np.arange(2), np.arange(2), np.arange(3)),
        determine_spatial_arr=lambda *a: None,
        determine_image=lambda *a: np.ones((2, 2)),
        determine_target_pixel_ifu=lambda **k: [1, 1],
        determine_spaxel_distances=lambda *a, **k: np.zeros((2, 2)),
        determine_reference_spectrum=lambda *a, **k: (np.ones(3), None, np.ones(3)),
        find_bars=lambda *a: list(range(30)),
        infer_fractional_position=lambda *a: (0.1, 0.1),
        dewiggle_lambda_space=fitter,
    )
    for name, value in replacements.items():
        monkeypatch.setattr(stack, name, value)
    for name in ["build_image_map_ifu", "build_reference_spectra_plot",
                 "build_bar_map_plot", "build_spaxel_map_plot_ifu"]:
        monkeypatch.setattr(stack, name, lambda *a, **k: None)

    stack.dewiggler_stack_ifu(
        "detector", "cube", str(tmp_path / "corrected.fits"), plot=False,
        n_jobs=1, use_cached_wcs=False, mask_bad_fits=False,
        target_wiggle_pixels=[[2, 1]], slices_to_use=[0, 1],
        try_different_slice_threshold=0, null_fit_improvement_threshold=threshold,
    )
    calls = fitter.call_args_list
    if threshold < 0:
        assert len(calls) == 2
        assert all("frequency_fudge" not in call.kwargs for call in calls)
    else:
        assert len(calls) == 4
        assert [call.kwargs["frequency_fudge"] for call in calls[::2]] == [2**0.5] * 2
        assert all("frequency_fudge" not in call.kwargs for call in calls[1::2])
    assert next(remaining, None) is None
    for call in calls:
        np.testing.assert_array_equal(call.args[3], original[:, 0, 1])
    expected = original.copy()
    expected[:, 0, 1] -= selected_offset
    assert len(written) == 1
    np.testing.assert_array_equal(written[0], expected)
