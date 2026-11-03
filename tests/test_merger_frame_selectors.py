import numpy as np
import pytest
from astropy.io import fits

from hyphen import frame_merger as merger


@pytest.fixture
def inputs(tmp_path):
    filenames = []
    for frame, value in enumerate((3., 7.)):
        path = tmp_path / f"frame-{frame}.fits"
        shape = (3, 2, 3)
        sci = fits.ImageHDU(np.full(shape, value), name="SCI")
        sci.header.update(CDELT3=0.01, CRVAL3=1.7)
        hdus = [fits.PrimaryHDU(), sci, fits.ImageHDU(np.ones(shape), name="ERR"),
                fits.ImageHDU(np.zeros(shape, dtype=np.uint32), name="DQ"),
                fits.ImageHDU(np.ones(shape), name="WMAP")]
        fits.HDUList(hdus).writeto(path)
        filenames.append([str(path)])
    return filenames


@pytest.mark.parametrize("selector, masked_frames", [(1, [1]), (None, [0, 1]), ("omitted", [0, 1])])
def test_valid_frame_masks_still_select_expected_frames(inputs, tmp_path, selector, masked_frames):
    entry = {"detector_idx": 0, "pixels": [[2, 1]]}
    if selector != "omitted":
        entry["frame_idx"] = selector
    mask = [entry]
    flux, _, _, weight, frames = merger.load_frames(inputs, str(tmp_path / "valid"), mask_config=mask)
    try:
        for frame in range(2):
            expected = 0 if frame in masked_frames else 1
            np.testing.assert_array_equal(weight[frame, :, 1, 2], expected)
            np.testing.assert_array_equal(flux[frame, :, 1, 2], expected * (3 + 4 * frame))
            np.testing.assert_array_equal(weight[frame, :, 0, 0], 1)
    finally:
        for group in frames:
            for hdul in group:
                hdul.close()


@pytest.mark.parametrize("invalid", [-1, 2])
def test_invalid_global_frame_selector_rejected_before_loader_io(inputs, tmp_path, monkeypatch, invalid):
    calls = []
    original_load, original_open = merger.np.load, merger.fits.open

    def load(*args, **kwargs):
        calls.append("cache")
        return original_load(*args, **kwargs)

    def open_fits(*args, **kwargs):
        calls.append("fits")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(merger.np, "load", load)
    monkeypatch.setattr(merger.fits, "open", open_fits)
    mask = [{"frame_idx": 0, "pixels": [[2, 1]]},
            {"frame_idx": invalid, "pixels": [[2, 1]]}]
    with pytest.raises(ValueError, match="frame_idx.*out of bounds"):
        merger.load_frames(inputs, str(tmp_path / "invalid"), mask_config=mask)
    assert calls == []
    assert not (tmp_path / "invalid_cached.npz").exists()
    assert not (tmp_path / "invalid_cached_v2.npz").exists()
    assert not (tmp_path / "invalid_cached_v3.npz").exists()


def test_invalid_frame_selector_prevents_final_output(inputs, tmp_path):
    output = tmp_path / "merged.fits"
    with pytest.raises(ValueError, match="frame_idx.*out of bounds"):
        merger.frame_merge(inputs, str(output), diagnostic_plot_filename="",
                           center_pixel_for_plots=[0, 0], target_detectors=["nrs1"],
                           mask_gap_union=False, mask_config=[{"frame_idx": 2, "pixels": [[2, 1]]}])
    assert not output.exists()
