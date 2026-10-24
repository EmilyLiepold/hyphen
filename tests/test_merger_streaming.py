from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

from hyphen.frame_merger import load_frames


def load_arrays(names, base, **kwargs):
    result = load_frames(names, str(base), **kwargs)
    try:
        return [result[i].copy() for i in (0, 2, 3)]  # SCI, DQ, WMAP; excludes ERR.
    finally:
        for frame in result[4]:
            for hdul in frame:
                hdul.close()


@pytest.mark.parametrize("all_missing", [False, True])
def test_streaming_detector_reduction_and_warm_cache(tmp_path, all_missing):
    shape = (2, 2, 3, 2, 4)  # frame, detector, wavelength, row, column
    values = np.arange(96, dtype=np.float32).reshape(shape)
    sci, weight = values / 8 + 1, (values % 5) / 4 + 1
    dq = np.zeros(shape, dtype=np.uint32)
    dq[0, 0, 0, 0, 1] = 1
    dq[:, 0, 1, 1, 3], dq[:, 1, 1, 1, 3] = 1, 2
    sci[0, 1, 0, 0, 2] = np.nan
    weight[0, 0, 2, 0, 3] = 0
    if all_missing:
        dq[:, 0], dq[:, 1] = 1, 2
    names = []
    for frame in range(2):
        names.append([])
        for detector in range(2):
            path = tmp_path / f"frame-{frame}-detector-{detector}.fits"
            hdus = [fits.PrimaryHDU()]
            for name, array in [("SCI", sci), ("ERR", np.ones(shape, dtype=np.float32)),
                                ("DQ", dq), ("WMAP", weight)]:
                hdus.append(fits.ImageHDU(array[frame, detector], name=name))
            fits.HDUList(hdus).writeto(path)
            names[-1].append(str(path))

    expected_sci = np.where(dq == 0, sci, np.nan)
    expected_weight = np.where(dq == 0, weight, np.nan)
    expected = [np.nansum(expected_sci, axis=1), np.minimum(dq[:, 0], dq[:, 1]),
                np.nansum(expected_weight, axis=1)]
    mask = [{"frame_idx": 1, "detector_idx": 1, "pixels": [[2, 1]]}]
    cold = load_arrays(names, tmp_path / "merge")
    with patch("hyphen.frame_merger.np.savez_compressed",
               side_effect=AssertionError("warm cache was rebuilt")):
        warm = load_arrays(names, tmp_path / "merge")
    for actual, cached, oracle in zip(cold, warm, expected):
        assert actual.dtype == cached.dtype == np.float32
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(oracle))
        np.testing.assert_array_equal(actual, oracle)
        np.testing.assert_array_equal(cached, actual)
    expected_sci[1, 1, :, 1, 2] = np.nan
    expected_weight[1, 1, :, 1, 2] = np.nan
    masked = load_arrays(names, tmp_path / "masked", mask_config=mask)
    masked_expected = [np.nansum(expected_sci, axis=1), expected[1],
                       np.nansum(expected_weight, axis=1)]
    for actual, oracle in zip(masked, masked_expected):
        np.testing.assert_array_equal(actual, oracle)
    if all_missing:
        for actual, scalar in zip(cold, [0, 1, 0]):
            np.testing.assert_array_equal(actual, np.full(actual.shape, scalar))
