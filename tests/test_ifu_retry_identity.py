"""A failed sampled IFU row must retry its original selected SCI spectrum."""
from types import SimpleNamespace

import numpy as np
import pytest

from hyphen import dewiggler_stack as stack


@pytest.mark.parametrize("fraction", [1, 2])
def test_subsampled_retry_preserves_target_identity(monkeypatch, tmp_path, fraction):
    original = np.arange(36., dtype=float).reshape(3, 3, 4) + 100
    targets = [(0, 1), (1, 3), (2, 0), (0, 2), (2, 3)]
    calls, written = [], []

    class Cube(list):
        def writeto(self, *args, **kwargs):
            written.append(self[1].data.copy())

    def load_frames(*args):
        cube = Cube([None, SimpleNamespace(data=original.copy()), None,
                     SimpleNamespace(data=np.zeros_like(original)),
                     SimpleNamespace(data=np.ones_like(original))])
        return [SimpleNamespace(data=np.ones((3, 4)))] * 8, cube, "detector", "cube"

    details = dict(alpha_order=0, beta_order=0, gamma_order=0, order=0)
    fft = dict(freq=np.array([1., 2.]), residual_fft=np.ones(2))

    def fit(*args, **kwargs):
        spectrum = args[3].copy()
        matches = [t for t in targets if np.array_equal(spectrum, original[:, t[0], t[1]])]
        assert len(matches) == 1, "every fit must consume an original selected SCI spectrum"
        target = matches[0]
        calls.append(target)
        bad = target == targets[2] and calls.count(target) == 1
        shift = 2 if bad else (5 if target == targets[2] else 1)
        return (np.ones(3), spectrum - shift, .04 if bad else .0001,
                np.arange(6.), details, fft)

    replacements = dict(
        load_frames=load_frames,
        determine_detector_name=lambda *a: "NRS1",
        get_wcs_ifu=lambda *a: (None, SimpleNamespace(pixel_to_world_values=lambda *a: a), None, None),
        get_2d_trace_model=lambda *a: lambda *b: (np.ones(3), np.ones(3)),
        get_principal_axes_ifu=lambda *a: (np.arange(4), np.arange(3), np.arange(3)),
        determine_spatial_arr=lambda *a: None,
        determine_image=lambda *a: np.ones((3, 4)),
        determine_target_pixel_ifu=lambda **k: [1, 1],
        determine_spaxel_distances=lambda *a, **k: np.zeros((4, 3)),
        determine_reference_spectrum=lambda *a, **k: (np.ones(3), None, np.ones(3)),
        find_bars=lambda *a: list(range(30)),
        infer_fractional_position=lambda *a: (0.1, 0.1),
        dewiggle_lambda_space=fit,
    )
    for name, value in replacements.items():
        monkeypatch.setattr(stack, name, value)
    for name in ["build_image_map_ifu", "build_reference_spectra_plot",
                 "build_bar_map_plot", "build_spaxel_map_plot_ifu"]:
        monkeypatch.setattr(stack, name, lambda *a, **k: None)

    stack.dewiggler_stack_ifu(
        "detector", "cube", str(tmp_path / "corrected.fits"), plot=False,
        n_jobs=1, use_cached_wcs=False, diagnostic_fraction=fraction,
        target_wiggle_pixels=[[col + 1, row + 1] for row, col in targets],
        slices_to_use=[0], null_fit_improvement_threshold=-1,
        fractional_residual_threshold=.1, power_ratio_threshold=2,
        mask_bad_fits=True,
    )
    assert calls == targets[::fraction] + [targets[2]]
    expected = original.copy()
    for row, col in targets[::fraction]:
        expected[:, row, col] -= 5 if (row, col) == targets[2] else 1
    assert len(written) == 1
    np.testing.assert_array_equal(written[0], expected)
