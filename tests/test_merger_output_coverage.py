import logging
from math import sqrt

import numpy as np
from astropy.io import fits

from hyphen.frame_merger import frame_merge


SHAPE = (5, 2, 3)


def write_cube(path, flux, sigma, weight, dq):
    sci = fits.ImageHDU(np.broadcast_to(np.asarray(flux)[:, None, None], SHAPE).copy(), name="SCI")
    sci.header.update(CDELT3=0.01, CRVAL3=1.7)
    err = fits.ImageHDU(np.full(SHAPE, sigma, dtype=np.float32), name="ERR")
    flags = fits.ImageHDU(np.broadcast_to(np.asarray(dq, dtype=np.uint32)[:, None, None], SHAPE).copy(), name="DQ")
    wmap = fits.ImageHDU(np.full(SHAPE, weight, dtype=np.float32), name="WMAP")
    fits.HDUList([fits.PrimaryHDU(), sci, err, flags, wmap]).writeto(path)


def test_no_coverage_and_surviving_exposures_cold_and_warm(tmp_path, caplog):
    high_bit = 1 << 31
    first = tmp_path / "first.fits"
    second = tmp_path / "second.fits"
    write_cube(first, [6, 6, 0, 6, 6], 2, 1, [high_bit | 4, 5, 0, 0, 0])
    write_cube(second, [10] * 5, 4, 3, [8, 0, 0, 8, 0])
    output = tmp_path / "merged.fits"
    cache = tmp_path / "merged_cached_v3.npz"
    caplog.set_level(logging.INFO, logger="hyphen.frame_merger")

    for phase in ("cold", "warm"):
        caplog.clear()
        frame_merge([[str(first)], [str(second)]], str(output),
                    diagnostic_plot_filename="", center_pixel_for_plots=[1, 0],
                    target_radius=0, mask_gap_union=False, target_detectors=["nrs1"],
                    nrs1_region=(0, 5), nrs2_region=(0, 5))
        loaded_from_disk = "Loading arrays from disk (frame-by-frame streaming)" in caplog.text
        assert loaded_from_disk == (phase == "cold")
        assert cache.exists()
        with fits.open(output) as product:
            for name, expected in (
                ("SCI", [np.nan, 10, 10, 6, 9]),
                ("ERR", [np.nan, 4, 4, 2, sqrt(148) / 4]),
                ("WMAP", [0, 3, 3, 1, 4]),
            ):
                np.testing.assert_allclose(product[name].data[:, 0, 0], expected, rtol=1e-6)
            dq = product["DQ"].data
            assert dq.dtype.kind == "u" and dq.dtype.itemsize == 4
            np.testing.assert_array_equal(dq[:, 0, 0], [high_bit | 5, 0, 0, 0, 0])


def test_rejected_detector_does_not_supply_values_or_dq(tmp_path):
    rejected = tmp_path / "rejected.fits"
    good = tmp_path / "good.fits"
    write_cube(rejected, [6] * 5, 2, 1, [4, 0, 0, 0, 0])
    write_cube(good, [10] * 5, 4, 3, [0] * 5)
    output = tmp_path / "merged.fits"
    frame_merge([[str(rejected), str(good)]], str(output), diagnostic_plot_filename="",
                center_pixel_for_plots=[1, 0], target_radius=0, mask_gap_union=False,
                target_detectors=["nrs1", "nrs2"], nrs1_region=(0, 5),
                nrs2_region=(0, 5))
    with fits.open(output) as product:
        for name, expected in (("SCI", 10), ("ERR", 4), ("WMAP", 3), ("DQ", 0)):
            np.testing.assert_allclose(product[name].data[0], expected)


def test_spatially_distinct_missing_coverage_causes(tmp_path):
    inputs = []
    for frame, flux in enumerate((6, 10)):
        path = tmp_path / f"frame-{frame}.fits"
        write_cube(path, [flux] * 5, 2, 1, [0] * 5)
        with fits.open(path, mode="update") as product:
            product["SCI"].data[0, 0, 0] = 0
            product["WMAP"].data[1, 0, 1] = 0
            product["WMAP"].data[2, 1, 0] = np.nan
        inputs.append([str(path)])

    output = tmp_path / "merged.fits"
    frame_merge(inputs, str(output), diagnostic_plot_filename="",
                center_pixel_for_plots=[1, 0], target_radius=0, mask_gap_union=False,
                target_detectors=["nrs1"], nrs1_region=(0, 5), nrs2_region=(0, 5))
    missing = [(0, 0, 0), (1, 0, 1), (2, 1, 0)]
    expected_sci = np.full(SHAPE, 8.)
    expected_err = np.full(SHAPE, sqrt(2))
    expected_weight = np.full(SHAPE, 2.)
    expected_dq = np.zeros(SHAPE, dtype=np.uint32)
    for index in missing:
        expected_sci[index] = np.nan
        expected_err[index] = np.nan
        expected_weight[index] = 0
        expected_dq[index] = 1
    with fits.open(output) as product:
        for name, expected in (("SCI", expected_sci), ("ERR", expected_err),
                               ("WMAP", expected_weight), ("DQ", expected_dq)):
            np.testing.assert_allclose(product[name].data, expected, rtol=1e-6)
