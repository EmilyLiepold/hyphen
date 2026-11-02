"""Replay must align saved rows with the current sampled IFU targets."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hyphen import dewiggler_stack as stack


@pytest.fixture
def replay(monkeypatch, tmp_path):
    original = np.arange(36., dtype=float).reshape(3, 3, 4) + 100
    targets = [(0, 1), (1, 3), (2, 0), (0, 2), (2, 3)]
    written, fits, traces = [], [], []

    class Cube(list):
        def writeto(self, *args, **kwargs):
            written.append(self[1].data.copy())

    def load_frames(*args):
        cube = Cube([None, SimpleNamespace(data=original.copy()), None,
                     SimpleNamespace(data=np.zeros_like(original)),
                     SimpleNamespace(data=np.ones_like(original))])
        return [SimpleNamespace(data=np.ones((3, 4)))] * 8, cube, "detector", "cube"

    details = dict(alpha_order=0, beta_order=0, gamma_order=0, order=0)
    fft = dict(freq=np.array([1., 2.]), spectrum_fft=np.ones(2), residual_fft=np.ones(2))

    def fit(*args, **kwargs):
        fits.append(args[3].copy())
        return np.ones(3), args[3] - 5, .0001, np.arange(6.), details, fft

    replacements = dict(
        load_frames=load_frames, determine_detector_name=lambda *a: "NRS1",
        get_wcs_ifu=lambda *a: (None, SimpleNamespace(pixel_to_world_values=lambda *x: x), None, None),
        get_2d_trace_model=lambda *a: lambda col, row: (col, row),
        get_principal_axes_ifu=lambda *a: (np.arange(4), np.arange(3), np.arange(3)),
        determine_spatial_arr=lambda *a: None, determine_image=lambda *a: np.ones((3, 4)),
        determine_target_pixel_ifu=lambda **k: [1, 1],
        determine_spaxel_distances=lambda *a, **k: np.zeros((4, 3)),
        determine_reference_spectrum=lambda *a, **k: (np.ones(3), None, np.ones(3)),
        find_bars=lambda *a: list(range(30)), infer_fractional_position=lambda *a: (0.1, 0.1),
        dewiggle_lambda_space=fit,
    )
    for name, value in replacements.items():
        monkeypatch.setattr(stack, name, value)

    def page(plot, *args, **kwargs):
        if plot:
            fig = plt.figure(figsize=(1, 1))
            plot.savefig(fig)
            plt.close(fig)

    def map_page(plot, *args, **kwargs):
        if plot:
            traces.append((tuple(args[6]), tuple(args[10][0])))
            page(plot)

    for name in ("build_image_map_ifu", "build_reference_spectra_plot", "build_bar_map_plot"):
        monkeypatch.setattr(stack, name, page)
    monkeypatch.setattr(stack, "build_spaxel_map_plot_ifu", map_page)
    monkeypatch.setattr(stack, "plot_dewiggle_result", lambda *a, **k: None)
    output = tmp_path / "cube.fits"

    def run(saved, current, fraction=2, plotted=True):
        rows = [(np.arange(6.), details, list(t), original[:, t[0], t[1]] - 1,
                 fft, .04 if t == targets[2] else .0001, "", 0) for t in saved]
        stack._save_fit_results(rows, str(output).replace(".fits", ".pkl"))
        return stack.dewiggler_stack_ifu(
            "detector", "cube", str(output), plot=str(tmp_path / "plots.pdf") if plotted else False,
            n_jobs=1, use_cached_wcs=False, diagnostic_fraction=fraction,
            target_wiggle_pixels=[[col + 1, row + 1] for row, col in current],
            slices_to_use=[0], null_fit_improvement_threshold=-1,
            fractional_residual_threshold=.1, power_ratio_threshold=2,
            mask_bad_fits=True, perform_dewiggling=False,
        )

    return SimpleNamespace(original=original, targets=targets, written=written,
                           fits=fits, traces=traces, run=run)


@pytest.mark.parametrize("unsampled_changed", [False, True], ids=["original", "changed"])
def test_aligned_replay_uses_saved_targets_for_plots_and_retry(replay, unsampled_changed):
    a, b, c, d, e = replay.targets
    current = [a, (0, 0), c, (1, 2), e] if unsampled_changed else [a, b, c, d, e]
    replay.run([a, c, e], current, fraction=np.int64(2))
    assert len(replay.fits) == 1
    np.testing.assert_array_equal(replay.fits[0], replay.original[:, c[0], c[1]])
    assert replay.traces == [(t, (t[1], t[0])) for t in (a, c, e)]
    expected = replay.original.copy()
    for row, col in (a, c, e):
        expected[:, row, col] -= 5 if (row, col) == c else 1
    assert len(replay.written) == 1
    np.testing.assert_array_equal(replay.written[0], expected)


@pytest.mark.parametrize("case", ["reordered", "missing", "extra", "saved_changed", "selection_changed"])
def test_mismatched_saved_pixels_stop_before_consumers(replay, case):
    a, b, c, d, e = replay.targets
    saved = dict(reordered=[c, a, e], missing=[a, e], extra=[a, c, e, b],
                 saved_changed=[a, b, e], selection_changed=[a, c, e])[case]
    current = [a, b, (1, 0), d, e] if case == "selection_changed" else replay.targets
    with pytest.raises(ValueError, match="saved coordinates must match the current sampled pixels"):
        replay.run(saved, current)
    assert not replay.traces and not replay.fits and not replay.written


@pytest.mark.parametrize("fraction", [0, -1, 1.5, True, np.bool_(True)],
                         ids=["zero", "negative", "float", "bool", "numpy-bool"])
def test_replay_rejects_unsafe_stride_before_consumers(replay, fraction):
    a, _, c, _, e = replay.targets
    with pytest.raises(ValueError, match="diagnostic_fraction must be a positive integer"):
        replay.run([a, c, e], replay.targets, fraction=fraction)
    assert not replay.traces and not replay.fits and not replay.written


@pytest.mark.parametrize("coordinate", [[1.0, 2], [True, 2], [1], [1, 2, 3], ["1", 2]],
                         ids=["float", "bool", "short", "long", "text"])
def test_replay_rejects_unsafe_coordinate_comparisons(coordinate):
    with pytest.raises(ValueError, match="saved coordinates must match the current sampled pixels"):
        stack._validate_ifu_replay_pixels([(None, None, coordinate)], np.array([[1], [2]]), 1)


def test_equal_duplicate_and_empty_sequences_are_not_rejected():
    row = (None, None, [0, 1])
    stack._validate_ifu_replay_pixels([row, row], np.array([[0, 0], [1, 1]]), 1)
    stack._validate_ifu_replay_pixels([], np.empty((2, 0), dtype=int), 1)
