import numpy as np
from astropy.io import fits

from hyphen.frame_merger import frame_merge


def test_explicit_detector_mask_excludes_its_uncertainty(tmp_path):
    inputs = []
    shape = (4, 2, 3)
    for detector, (flux, sigma) in enumerate(((3., 1.), (7., 2.))):
        path = tmp_path / f"detector-{detector}.fits"
        sci = fits.ImageHDU(np.full(shape, flux), name="SCI")
        sci.header.update(CDELT3=0.01, CRVAL3=1.7)
        hdus = [fits.PrimaryHDU(), sci, fits.ImageHDU(np.full(shape, sigma), name="ERR"),
                fits.ImageHDU(np.zeros(shape, dtype=np.uint32), name="DQ"),
                fits.ImageHDU(np.ones(shape), name="WMAP")]
        fits.HDUList(hdus).writeto(path)
        inputs.append(str(path))

    output = tmp_path / "merged.fits"
    frame_merge([inputs], str(output), diagnostic_plot_filename="",
                center_pixel_for_plots=[1, 0], target_radius=0,
                nrs1_region=(0, 4), nrs2_region=(0, 4), mask_gap_union=False,
                target_detectors=["nrs1", "nrs2"],
                mask_config=[{"frame_idx": 0, "detector_idx": 0, "pixels": [[2, 1]]}])
    with fits.open(output) as product:
        for name, masked, neighbor in (("SCI", 7., 10.), ("WMAP", 1., 2.)):
            np.testing.assert_array_equal(product[name].data[:, 1, 2], masked)
            np.testing.assert_array_equal(product[name].data[:, 1, 1], neighbor)
        np.testing.assert_allclose(product["ERR"].data[:, 1, 2], 2., rtol=1e-6)
        np.testing.assert_allclose(product["ERR"].data[:, 1, 1], np.sqrt(5.), rtol=1e-6)
