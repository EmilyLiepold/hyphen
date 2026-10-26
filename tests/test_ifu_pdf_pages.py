"""Check actual IFU PDF assembly using labeled pages and synthetic replay rows."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pypdf import PdfReader
import pytest

from hyphen import dewiggler_stack as stack


def labeled_page(pdf, label):
    fig = plt.figure(figsize=(4, 3))
    fig.text(0.1, 0.5, label)
    pdf.savefig(fig)
    plt.close(fig)


@pytest.mark.parametrize("supplied_pdf", [False, True])
def test_ifu_keeps_overview_and_orders_all_chunk_pages(monkeypatch, tmp_path, supplied_pdf):
    data = np.ones((3, 4, 4))
    hdus = [None, SimpleNamespace(data=data.copy()), None,
            SimpleNamespace(data=np.zeros_like(data)), SimpleNamespace(data=data)]
    cube = Mock()
    cube.__getitem__ = Mock(side_effect=hdus.__getitem__)
    detector = [SimpleNamespace(data=np.ones((4, 4)))] * 8
    distance = np.hypot(*np.meshgrid(np.arange(4) * 0.4, np.arange(4) * 0.4))
    positions = [(i // 4, i % 4) for i in range(11)]
    fft = {"freq": np.array([1.]), "residual_fft": np.array([1.])}
    rows = [(None, {}, pos, np.ones(3), fft, 0.0001, "", 0) for pos in positions]
    replacements = dict(
        load_frames=lambda *a: (detector, cube, "detector", "PDF-FIXTURE"),
        determine_detector_name=lambda *a: "NRS1",
        get_wcs_ifu=lambda *a: (None, SimpleNamespace(pixel_to_world_values=lambda *a: a), None, None),
        get_2d_trace_model=lambda *a: None,
        get_principal_axes_ifu=lambda *a: (np.arange(4) / 3600, np.arange(4) / 3600, np.arange(3)),
        determine_spatial_arr=lambda *a: None,
        determine_image=lambda *a: np.ones((4, 4)),
        determine_target_pixel_ifu=lambda **k: [1, 1],
        determine_spaxel_distances=lambda *a, **k: distance,
        determine_reference_spectrum=lambda *a, **k: (np.ones(3), None, np.ones(3)),
        find_bars=lambda *a: list(range(30)),
        _load_fit_results=lambda *a: rows,
        build_reference_spectra_plot=lambda pdf, *a, **k: labeled_page(pdf, "REFERENCES"),
        build_bar_map_plot=lambda pdf, *a, **k: labeled_page(pdf, "BARS"),
    )
    for name, value in replacements.items():
        monkeypatch.setattr(stack, name, value)

    completed = []

    class SerialReversed:
        def __init__(self, **kwargs):
            pass

        def __call__(self, jobs):
            result = [function(*args, **kwargs) for function, args, kwargs in jobs]
            return result[::-1]

    def chunk(**kwargs):
        index = kwargs["chunk_idx"]
        path = Path(kwargs["tmpdir"]) / f"chunk_{index}.pdf"
        with PdfPages(path) as pdf:
            labeled_page(pdf, f"SPAXEL-{index:02d}")
            if index == 0:
                labeled_page(pdf, "SPAXEL-00-EXTRA")
        completed.append(index)
        return str(path), ""

    monkeypatch.setattr(stack, "ProgressParallel", SerialReversed)
    monkeypatch.setattr(stack, "_plot_chunk_worker", chunk)
    destination = tmp_path / "diagnostics.pdf"
    plot = PdfPages(destination) if supplied_pdf else str(destination)
    if supplied_pdf:
        labeled_page(plot, "PREEXISTING")
    try:
        stack.dewiggler_stack_ifu(
            "synthetic-detector", "synthetic-cube", str(tmp_path / "unused.fits"),
            plot=plot, perform_dewiggling=False, n_jobs=11, use_cached_wcs=False,
            target_wiggle_pixels=[[col + 1, row + 1] for row, col in positions],
            mask_bad_fits=False,
        )
    finally:
        if supplied_pdf:
            plot.close()
    assert completed == list(range(11))
    texts = [page.extract_text() for page in PdfReader(destination).pages]
    expected = (["PREEXISTING"] if supplied_pdf else []) + [
        "Median image of PDF-FIXTURE", "REFERENCES", "BARS", "SPAXEL-00",
        "SPAXEL-00-EXTRA", *[f"SPAXEL-{i:02d}" for i in range(1, 11)],
    ]
    assert len(texts) == len(expected)
    for text, identity in zip(texts, expected):
        if identity.startswith("Median image"):
            assert identity in text
        else:
            assert text.strip() == identity
    cube.writeto.assert_called_once_with(str(tmp_path / "unused.fits"), overwrite=True)
