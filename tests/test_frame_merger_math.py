from math import sqrt

from astropy.io import fits
import numpy as np
import pytest

from hyphen.frame_merger import frame_merge


SHAPE = (4, 2, 3)  # wavelength, slice, pixel


def _write_cube(path, flux, sigma, weight):
    primary = fits.PrimaryHDU()
    primary.header.update(DATAMODL="IFUCubeModel", ERREXT="ERR")
    sci = fits.ImageHDU(np.full(SHAPE, flux, dtype=np.float32), name="SCI")
    sci.header.update(BUNIT="MJy/sr", CDELT3=0.01, CRVAL3=1.7, CRPIX3=1.0)
    err = fits.ImageHDU(np.full(SHAPE, sigma, dtype=np.float32), name="ERR")
    err.header.update(BUNIT="MJy/sr", ERRTYPE="ERR")
    dq = fits.ImageHDU(np.zeros(SHAPE, dtype=np.uint32), name="DQ")
    wmap = fits.ImageHDU(np.full(SHAPE, weight, dtype=np.float32), name="WMAP")
    fits.HDUList([primary, sci, err, dq, wmap]).writeto(path)


@pytest.mark.parametrize(
    "sigmas,weights", [((2, 2), (1, 1)), ((2, 4), (1, 3))],
    ids=["equal-weights", "unequal-weights"],
)
def test_merger_propagates_independent_standard_deviations(tmp_path, sigmas, weights):
    fluxes = (6, 10)
    filenames = []
    for i, (flux, sigma, weight) in enumerate(zip(fluxes, sigmas, weights)):
        path = tmp_path / f"frame-{i}.fits"
        _write_cube(path, flux, sigma, weight)
        filenames.append([str(path)])
    output = tmp_path / "merged.fits"
    frame_merge(
        filenames, str(output), diagnostic_plot_filename="",
        center_pixel_for_plots=[1, 0], target_radius=0,
        mask_gap_union=False, target_detectors=["nrs1"],
        nrs1_region=(0, 4), nrs2_region=(0, 4), load_frames_from_disk=True,
    )
    total_weight = sum(weights)
    expected_flux = sum(w * f for w, f in zip(weights, fluxes)) / total_weight
    expected_sigma = sqrt(sum(w**2 * s**2 for w, s in zip(weights, sigmas))) / total_weight
    with fits.open(output) as product:
        assert product[0].header["NFRAMES"] == 2
        assert product[0].header["ERREXT"] == "ERR"
        assert product["ERR"].header["ERRTYPE"] == "ERR"
        assert product["SCI"].header["BUNIT"] == "MJy/sr"
        assert product["ERR"].header["BUNIT"] == "MJy/sr"
        for name in ("SCI", "ERR", "WMAP"):
            assert product[name].data.shape == SHAPE
            assert np.all(np.isfinite(product[name].data))
        np.testing.assert_allclose(product["SCI"].data, expected_flux, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(product["WMAP"].data, total_weight, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(product["ERR"].data, expected_sigma, rtol=1e-6, atol=1e-7)


def test_merger_variance_cache_rejects_legacy_and_reuses_warm(tmp_path, monkeypatch):
    filenames = []
    for frame in range(2):
        detectors = []
        for detector, sigma in enumerate((1, 2)):
            path = tmp_path / f"frame-{frame}-detector-{detector}.fits"
            _write_cube(path, 3, sigma, 0.5)
            detectors.append(str(path))
        filenames.append(detectors)
    output = tmp_path / "merged.fits"
    legacy = tmp_path / "merged_cached.npz"
    poison = np.full((2, *SHAPE), 99, dtype=np.float32)
    np.savez_compressed(legacy, **dict.fromkeys(("fluxes", "variances", "dqs", "weights"), poison))
    legacy_bytes = legacy.read_bytes()
    options = dict(diagnostic_plot_filename="", center_pixel_for_plots=[1, 0],
                   target_radius=0, mask_gap_union=False, target_detectors=["nrs1", "nrs2"],
                   nrs1_region=(0, 4), nrs2_region=(0, 4), load_frames_from_disk=False)
    original_open = fits.open

    def template_only(path, *args, **kwargs):
        assert str(path) == filenames[0][0], "Warm cache must not reload detector arrays"
        return original_open(path, *args, **kwargs)

    for phase in ("cold", "warm"):
        with monkeypatch.context() as patch:
            if phase == "warm":
                patch.setattr(fits, "open", template_only)
            frame_merge(filenames, str(output), **options)
        with fits.open(output) as product:
            for name, expected in (("SCI", 6), ("ERR", sqrt(2.5)), ("WMAP", 2)):
                assert product[name].data.shape == SHAPE
                assert np.all(np.isfinite(product[name].data))
                np.testing.assert_allclose(product[name].data, expected, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(product["DQ"].data, 0)
            assert product["SCI"].header["CRPIX3"] == 1.0
            assert product["SCI"].header["CRVAL3"] == 1.7
            assert product["SCI"].header["CDELT3"] == 0.01
        with np.load(tmp_path / "merged_cached_v2.npz") as cache:
            assert cache["variances"].dtype == np.float32
            np.testing.assert_allclose(cache["variances"], 5, rtol=1e-6, atol=1e-7)
        assert legacy.read_bytes() == legacy_bytes
