"""NaN ERR excludes a detector sample before detector reduction."""

from math import sqrt

import numpy as np
from astropy.io import fits
from stdatamodels.jwst.datamodels import dqflags

from hyphen.frame_merger import frame_merge, load_frames


def disjoint_cubes(tmp_path, inconsistent):
    shape = (6, 2, 3)
    inputs = []
    for frame, (flux, sigma, weight) in enumerate(((6., 2., 1.), (10., 4., 3.))):
        pair = []
        for detector in range(2):
            valid = np.zeros(shape, dtype=bool)
            valid[detector * 3:(detector + 1) * 3] = True
            sci = np.where(valid, flux, np.nan)
            err = np.where(valid, sigma, np.nan)
            wmap = np.where(valid, weight, 0.)
            if inconsistent:
                if frame == 1 and detector == 0:
                    err[1] = np.nan
                if detector == 1:
                    err[4] = np.nan
            sci_hdu = fits.ImageHDU(sci, name="SCI")
            sci_hdu.header.update(CDELT3=0.01, CRVAL3=1.7)
            path = tmp_path / f"frame-{frame}-detector-{detector}.fits"
            fits.HDUList([fits.PrimaryHDU(), sci_hdu, fits.ImageHDU(err, name="ERR"),
                          fits.ImageHDU(np.zeros(shape, dtype=np.uint32), name="DQ"),
                          fits.ImageHDU(wmap, name="WMAP")]).writeto(path)
            pair.append(str(path))
        inputs.append(pair)
    return inputs


def merge(inputs, output, gap=False):
    frame_merge(inputs, str(output), diagnostic_plot_filename="", center_pixel_for_plots=[1, 0],
                target_radius=0, mask_gap_union=gap, gap_region=[0, 5],
                target_detectors=["nrs1", "nrs2"], nrs1_region=(0, 3), nrs2_region=(3, 6))
    with fits.open(output) as hdus:
        return {name: hdus[name].data[:, 0, 0].copy() for name in ("SCI", "ERR", "WMAP", "DQ")}


def test_disjoint_nan_err_and_legacy_frame_cache(tmp_path):
    inputs = disjoint_cubes(tmp_path, inconsistent=True)
    output = tmp_path / "merged.fits"
    legacy = tmp_path / "merged_cached_v2.npz"
    shape = (2, 6, 2, 3)
    np.savez_compressed(legacy, fluxes=np.full(shape, 99.), variances=np.ones(shape),
                        dqs=np.zeros(shape), weights=np.ones(shape))
    old = legacy.read_bytes()
    cold = merge(inputs, output)
    assert legacy.read_bytes() == old
    assert (tmp_path / "merged_cached_v3.npz").exists()
    warm = merge(inputs, output)
    for name in cold:
        np.testing.assert_array_equal(warm[name], cold[name])
    np.testing.assert_allclose(cold["SCI"], [9, 6, 9, 9, np.nan, 9], rtol=1e-6)
    np.testing.assert_allclose(cold["ERR"], [sqrt(148)/4, 2, sqrt(148)/4, sqrt(148)/4, np.nan, sqrt(148)/4], rtol=1e-6)
    np.testing.assert_array_equal(cold["WMAP"], [4, 1, 4, 4, 0, 4])
    np.testing.assert_array_equal(cold["DQ"], [0, 0, 0, 0, dqflags.pixel["DO_NOT_USE"], 0])


def test_disjoint_baseline_and_legacy_gap_cache(tmp_path):
    inputs = disjoint_cubes(tmp_path, inconsistent=False)
    output = tmp_path / "merged.fits"
    legacy = tmp_path / "merged_gaps.npy"
    np.save(legacy, np.full((3, 2, 2, 2), 1.))
    old = legacy.read_bytes()
    cold = merge(inputs, output, gap=True)
    assert legacy.read_bytes() == old
    assert (tmp_path / "merged_gaps_v3.npy").exists()
    warm = merge(inputs, output, gap=True)
    for name in cold:
        np.testing.assert_array_equal(warm[name], cold[name])
    np.testing.assert_allclose(cold["SCI"], np.full(6, 9.), rtol=1e-6)
    np.testing.assert_allclose(cold["ERR"], np.full(6, sqrt(148)/4), rtol=1e-6)
    np.testing.assert_array_equal(cold["WMAP"], np.full(6, 4.))
    np.testing.assert_array_equal(cold["DQ"], np.zeros(6, dtype=np.uint32))


def test_finite_zero_negative_and_infinite_err_are_unchanged(tmp_path):
    shape = (4, 1, 1)
    path = tmp_path / "values.fits"
    sci = fits.ImageHDU(np.ones(shape), name="SCI")
    fits.HDUList([fits.PrimaryHDU(), sci,
                  fits.ImageHDU(np.array([0., -2., np.inf, -np.inf]).reshape(shape), name="ERR"),
                  fits.ImageHDU(np.zeros(shape, dtype=np.uint32), name="DQ"),
                  fits.ImageHDU(np.ones(shape), name="WMAP")]).writeto(path)
    flux, variance, dq, weight, frames = load_frames([[str(path)]], str(tmp_path / "values"))
    try:
        np.testing.assert_array_equal(flux[0, :, 0, 0], np.ones(4))
        np.testing.assert_array_equal(weight[0, :, 0, 0], np.ones(4))
        np.testing.assert_array_equal(variance[0, :, 0, 0], [0, 4, np.inf, np.inf])
    finally:
        frames[0][0].close()
