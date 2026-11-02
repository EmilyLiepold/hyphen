"""Cache-use warnings leave the existing cache and reload behavior intact."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest
from astropy.io import fits

from hyphen import frame_merger


RAW_WARNING = "Using cached frame arrays"
GAP_WARNING = "Using cached gap edges"


def _fits_file(path):
    data = np.ones((6, 1, 1), dtype=np.float32)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data, name="SCI"),
        fits.ImageHDU(data, name="ERR"),
        fits.ImageHDU(np.zeros_like(data, dtype=np.uint32), name="DQ"),
        fits.ImageHDU(data, name="WMAP"),
    ]).writeto(path)


def _load_arrays(filename, base, **kwargs):
    result = frame_merger.load_frames([[str(filename)]], str(base), **kwargs)
    try:
        return [array.copy() for array in result[:4]]
    finally:
        for frame in result[4]:
            for hdul in frame:
                hdul.close()


def test_raw_cache_hit_and_forced_reload(tmp_path, caplog):
    filename = tmp_path / "input.fits"
    _fits_file(filename)
    base = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger=frame_merger.__name__):
        cold = _load_arrays(filename, base)
        assert RAW_WARNING not in caplog.text
        with fits.open(filename, mode="update") as hdul:
            hdul[1].data += 5
        caplog.clear()
        warm = _load_arrays(filename, base)
        assert RAW_WARNING in caplog.text
        caplog.clear()
        forced = _load_arrays(filename, base, load_frames_from_disk=True)
        assert RAW_WARNING not in caplog.text
        with fits.open(filename, mode="update") as hdul:
            hdul[1].data += 3
        caplog.clear()
        masked = _load_arrays(filename, base, mask_config=[])
        assert RAW_WARNING not in caplog.text
    for a, b in zip(warm, cold):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(forced[0], cold[0] + 5)
    np.testing.assert_array_equal(masked[0], forced[0] + 3)
    for actual in (forced, masked):
        for a, b in zip(actual[1:], cold[1:]):
            np.testing.assert_array_equal(a, b)


def test_gap_cache_hit_and_forced_reload(tmp_path, monkeypatch, caplog):
    base = tmp_path / "out"
    weights = np.ones((1, 6, 1, 1))
    calculate = Mock(side_effect=[(1, 3), (2, 4)])
    monkeypatch.setattr(frame_merger, "identify_chip_gap", calculate)
    with caplog.at_level(logging.WARNING, logger=frame_merger.__name__):
        cold = frame_merger.determine_gap_edges(weights, str(base))
        assert GAP_WARNING not in caplog.text
        assert calculate.call_count == 1
        caplog.clear()
        warm = frame_merger.determine_gap_edges(weights, str(base))
        assert GAP_WARNING in caplog.text
        assert calculate.call_count == 1
        caplog.clear()
        forced = frame_merger.determine_gap_edges(weights, str(base), recompute_gaps=True)
        assert GAP_WARNING not in caplog.text
        assert calculate.call_count == 2
    np.testing.assert_array_equal(warm, cold)
    assert not np.array_equal(forced, cold)


@pytest.mark.parametrize("force", [False, True])
def test_malformed_caches_still_raise_before_forced_reload(tmp_path, force):
    raw_base = tmp_path / "raw"
    # The baseline and assembled candidate use different raw-cache suffixes.
    for suffix in ("_cached.npz", "_cached_v2.npz"):
        np.savez_compressed(f"{raw_base}{suffix}", fluxes=[1])
    with pytest.raises(KeyError, match="variances"):
        frame_merger.load_frames([["unused.fits"]], str(raw_base), load_frames_from_disk=force)
    gap_base = tmp_path / "gap"
    (tmp_path / "gap_gaps.npy").write_bytes(b"not a NumPy file")
    with pytest.raises(ValueError):
        frame_merger.determine_gap_edges(np.ones((1, 6, 1, 1)), str(gap_base), recompute_gaps=force)


def test_yaml_dispatch_writes_raw_cache_warning_to_merger_log(tmp_path, monkeypatch):
    filename = tmp_path / "input_f1_nrs1.fits"
    _fits_file(filename)
    base = tmp_path / "merged"
    data = np.ones((1, 6, 1, 1), dtype=np.float32)
    for suffix in ("_cached.npz", "_cached_v2.npz"):
        np.savez_compressed(f"{base}{suffix}", fluxes=data, variances=data, dqs=data, weights=data)
    config = tmp_path / "config.yaml"
    config.write_text(
        "step_type: merge\nlabel: test\n"
        f"input_filename_prototype: {tmp_path / 'input_FRAME_DETECTOR.fits'}\n"
        f"output_filename: {base}.fits\n"
        "target_frames: [f1]\ntarget_detectors: [nrs1]\n"
    )
    def fake_merge(names, output, **kwargs):
        _load_arrays(filename, base)
    monkeypatch.setattr(frame_merger, "frame_merge", fake_merge)
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    root.handlers = []
    try:
        frame_merger.frame_merge_wrapper(str(config))
        for handler in root.handlers:
            handler.flush()
        assert RAW_WARNING in (tmp_path / "merged.log").read_text()
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers = original_handlers
        root.setLevel(original_level)
