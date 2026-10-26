"""Exercise plotted first-pass fitting, actual sidecars, then unplotted replay."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from hyphen import dewiggler_stack as stack


def diagnostic_page(pdf, *args, **kwargs):
    if pdf:
        fig = plt.figure(figsize=(3, 2))
        fig.text(0.1, 0.5, "SAVED-REPLAY-FIXTURE")
        pdf.savefig(fig)
        plt.close(fig)


def test_ifu_plotted_fit_saves_and_unplotted_replay_uses_pair(monkeypatch, tmp_path):
    original = np.arange(12., dtype=float).reshape(3, 2, 2) + 10
    written, loaded, rendered = [], [], []

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
    fitter = Mock(side_effect=lambda *a, **k: (
        np.ones(3), a[3] - 2, 0.0001, np.arange(6.), details, fft))
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
        monkeypatch.setattr(stack, name, diagnostic_page)

    def chunk(**kwargs):
        rendered.extend(row[2] for row in kwargs["xsols_chunk"])
        path = Path(kwargs["tmpdir"]) / f"chunk_{kwargs['chunk_idx']}.pdf"
        with PdfPages(path) as pdf:
            diagnostic_page(pdf)
        return str(path), ""

    save = Mock(wraps=stack._save_fit_results)
    real_load = stack._load_fit_results

    def observe_load(base):
        rows = real_load(base)
        loaded.append(rows)
        return rows

    monkeypatch.setattr(stack, "_save_fit_results", save)
    monkeypatch.setattr(stack, "_load_fit_results", observe_load)
    monkeypatch.setattr(stack, "_plot_chunk_worker", chunk)
    output = tmp_path / "corrected.fits"
    options = dict(n_jobs=1, use_cached_wcs=False, mask_bad_fits=False,
                   target_wiggle_pixels=[[1, 1], [2, 2]], null_fit_improvement_threshold=-1)
    stack.dewiggler_stack_ifu("detector", "cube", str(output),
                             plot=str(tmp_path / "fit.pdf"), **options)
    assert fitter.call_count == 2
    save.assert_called_once()
    assert loaded == [] and rendered == [[0, 0], [1, 1]]
    base = str(output).replace(".fits", ".pkl")
    pair = [Path(base + suffix) for suffix in [".npz", ".json"]]
    assert all(p.is_file() for p in pair)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in pair]
    fitter.reset_mock(side_effect=True)
    fitter.side_effect = AssertionError("replay must not invoke the fitter")
    stack.dewiggler_stack_ifu("detector", "cube", str(output),
                             plot=False, perform_dewiggling=False, **options)
    fitter.assert_not_called()
    assert save.call_count == 1 and len(loaded) == 1
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in pair] == before
    assert rendered == [[0, 0], [1, 1]]
    saved = save.call_args.args[0]
    assert len(saved) == len(loaded[0]) == 2
    for first, replay in zip(saved, loaded[0]):
        for index in [0, 3]:
            np.testing.assert_array_equal(replay[index], first[index])
        for index in [1, 2, 5, 6, 7]:
            assert replay[index] == first[index]
        assert replay[4].keys() == first[4].keys()
        for key in first[4]:
            np.testing.assert_array_equal(replay[4][key], first[4][key])
    expected = original.copy()
    expected[:, 0, 0] -= 2
    expected[:, 1, 1] -= 2
    assert len(written) == 2
    for data in written:
        np.testing.assert_array_equal(data, expected)
