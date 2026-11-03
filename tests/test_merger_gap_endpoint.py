import numpy as np
import pytest
from astropy.io import fits

from hyphen.frame_merger import frame_merge, identify_chip_gap


@pytest.mark.parametrize(
    "weights, expected", [([1, 0, 0, 1], (1, 2)), ([0, 1, 1, 1], (0, 0)),
                          ([1, 1, 1, 0], (3, 3))],
)
def test_gap_finder_returns_inclusive_edges(weights, expected):
    assert identify_chip_gap(np.array(weights, dtype=float)) == expected


@pytest.mark.parametrize(
    "edges, masked",
    [([(1, 2), (2, 3)], [1, 2, 3]),
     ([(0, 0), (0, 0)], [0]),
     ([(2, 2), (2, 2)], [2]),
     ([(4, 4), (4, 4)], [4]),
     ([(np.nan, np.nan), (np.nan, np.nan)], []),
     ([(np.nan, 3), (1, np.nan)], [])],
    ids=["different-endpoints", "first-channel", "single-channel",
         "last-channel", "no-valid-pair", "partial-pairs"],
)
def test_gap_union_masks_both_inclusive_edges(tmp_path, edges, masked):
    inputs = []
    for frame, flux in enumerate((6., 10.)):
        path = tmp_path / f"frame-{frame}.fits"
        primary = fits.PrimaryHDU()
        sci = fits.ImageHDU(np.full((5, 1, 1), flux), name="SCI")
        sci.header.update(CDELT3=0.01, CRVAL3=1.7)
        err = fits.ImageHDU(np.ones((5, 1, 1)), name="ERR")
        dq = fits.ImageHDU(np.zeros((5, 1, 1), dtype=np.uint32), name="DQ")
        wmap = fits.ImageHDU(np.ones((5, 1, 1)), name="WMAP")
        fits.HDUList([primary, sci, err, dq, wmap]).writeto(path)
        inputs.append([str(path)])

    output = tmp_path / "merged.fits"
    cache = tmp_path / "merged_gaps.npy"
    np.save(cache, np.array(edges, dtype=float).reshape(1, 1, 2, 2))
    original_cache = cache.read_bytes()
    frame_merge(inputs, str(output), diagnostic_plot_filename="",
                center_pixel_for_plots=[0, 0], target_radius=0,
                load_frames_from_disk=True, mask_gap_union=True,
                gap_region=[0, 4], target_detectors=["nrs1"],
                nrs1_region=(0, 5), nrs2_region=(0, 5))

    expected_weight = np.full(5, 2.)
    expected_flux = np.full(5, 8.)
    expected_weight[masked] = 0
    expected_flux[masked] = np.nan
    with fits.open(output) as product:
        np.testing.assert_array_equal(product["WMAP"].data[:, 0, 0], expected_weight)
        np.testing.assert_array_equal(product["SCI"].data[:, 0, 0], expected_flux)
    assert cache.read_bytes() == original_cache
