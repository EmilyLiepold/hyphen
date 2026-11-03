import numpy as np
import pytest
from astropy.io import fits

from hyphen.frame_merger import frame_merge, identify_chip_gap


@pytest.mark.parametrize("weights", [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0],
                                     [np.nan] * 5, [-1] * 5])
def test_no_gap_or_no_coverage_has_no_edges(weights):
    assert np.isnan(identify_chip_gap(np.asarray(weights, dtype=float))).all()


def test_real_single_channel_gap_keeps_inclusive_edges():
    assert identify_chip_gap(np.array([1, 1, 0, 1, 1.])) == (2, 2)


def write_cube(path, flux, weights):
    shape = (5, 1, 1)
    primary = fits.PrimaryHDU()
    sci = fits.ImageHDU(np.full(shape, flux), name="SCI")
    sci.header.update(CDELT3=0.01, CRVAL3=1.7)
    err = fits.ImageHDU(np.ones(shape), name="ERR")
    dq = fits.ImageHDU(np.zeros(shape, dtype=np.uint32), name="DQ")
    wmap = fits.ImageHDU(np.asarray(weights, dtype=float).reshape(shape), name="WMAP")
    fits.HDUList([primary, sci, err, dq, wmap]).writeto(path)


def merge_pair(tmp_path, first_weights, second_weights, cached_edges=None):
    inputs = []
    for frame, (flux, weights) in enumerate(((6., first_weights), (10., second_weights))):
        path = tmp_path / f"frame-{frame}.fits"
        write_cube(path, flux, weights)
        inputs.append([str(path)])
    output = tmp_path / "merged.fits"
    cache = tmp_path / "merged_gaps_v3.npy"
    if cached_edges is not None:
        np.save(cache, np.asarray(cached_edges, dtype=float).reshape(1, 1, 2, 2))

    results = []
    for _ in range(2):
        frame_merge(inputs, str(output), diagnostic_plot_filename="",
                    center_pixel_for_plots=[0, 0], target_radius=0,
                    load_frames_from_disk=True, mask_gap_union=True,
                    gap_region=[0, 4], target_detectors=["nrs1"],
                    nrs1_region=(0, 5), nrs2_region=(0, 5))
        with fits.open(output) as product:
            results.append((product["SCI"].data[:, 0, 0].copy(),
                            product["WMAP"].data[:, 0, 0].copy()))
    for actual, warm in zip(results[0], results[1]):
        np.testing.assert_array_equal(actual, warm)
    return results[0], np.load(cache)


@pytest.mark.parametrize(
    "first, second, expected_sci, expected_weight",
    [([1] * 5, [1] * 5, [8] * 5, [2] * 5),
     ([1, 1, 0, 1, 1], [1] * 5, [8, 8, np.nan, 8, 8], [2, 2, 0, 2, 2]),
     ([0] * 5, [1] * 5, [10] * 5, [1] * 5),
     ([np.nan] * 5, [1] * 5, [10] * 5, [1] * 5),
     ([1, 0, 1, 1, 1], [1, 1, 1, 0, 1], [8, np.nan, np.nan, np.nan, 8], [2, 0, 0, 0, 2])],
    ids=["no-gap", "one-channel-gap", "zero-weight-frame", "nan-weight-frame",
         "different-gaps"],
)
def test_fresh_gap_and_warm_cache(tmp_path, first, second, expected_sci, expected_weight):
    (sci, weight), _ = merge_pair(tmp_path, first, second)
    np.testing.assert_array_equal(sci, expected_sci)
    np.testing.assert_array_equal(weight, expected_weight)


def test_legacy_full_span_cache_does_not_expand_missing_frame(tmp_path):
    (sci, weight), edges = merge_pair(tmp_path, [0] * 5, [1, 1, 0, 1, 1],
                                      [(0, 4), (2, 2)])
    np.testing.assert_array_equal(sci, [10, 10, np.nan, 10, 10])
    np.testing.assert_array_equal(weight, [1, 1, 0, 1, 1])
    np.testing.assert_array_equal(edges.reshape(2, 2), [(0, 4), (2, 2)])


@pytest.mark.parametrize("partial", [(np.nan, 4), (0, np.nan)])
def test_partial_cached_pair_does_not_expand_complete_gap(tmp_path, partial):
    (sci, weight), _ = merge_pair(tmp_path, [1] * 5, [1] * 5,
                                  [(2, 2), partial])
    np.testing.assert_array_equal(sci, [8, 8, np.nan, 8, 8])
    np.testing.assert_array_equal(weight, [2, 2, 0, 2, 2])


@pytest.mark.parametrize("invalid", [(2, 9), (2, 3.5)])
def test_out_of_array_or_fractional_pair_does_not_expand_union(tmp_path, invalid):
    (sci, weight), _ = merge_pair(tmp_path, [1] * 5, [1] * 5,
                                  [(2, 2), invalid])
    np.testing.assert_array_equal(sci, [8, 8, np.nan, 8, 8])
    np.testing.assert_array_equal(weight, [2, 2, 0, 2, 2])


def test_non_square_spatial_grid_cold_and_warm_gap_cache(tmp_path):
    shape = (5, 2, 3)
    weights = np.ones((2, *shape))
    weights[0, 2, 0, 1] = 0
    weights[0, :, 0, 2] = 0
    weights[0, 1, 1, 0] = 0
    weights[1, 3, 1, 0] = 0
    weights[1, 4, 1, 1] = 0
    weights[0, 0, 1, 2] = 0
    inputs = []
    for frame, flux in enumerate((6., 10.)):
        primary = fits.PrimaryHDU()
        sci = fits.ImageHDU(np.full(shape, flux), name="SCI")
        sci.header.update(CDELT3=0.01, CRVAL3=1.7)
        err = fits.ImageHDU(np.ones(shape), name="ERR")
        dq = fits.ImageHDU(np.zeros(shape, dtype=np.uint32), name="DQ")
        wmap = fits.ImageHDU(weights[frame], name="WMAP")
        path = tmp_path / f"frame-{frame}.fits"
        fits.HDUList([primary, sci, err, dq, wmap]).writeto(path)
        inputs.append([str(path)])

    expected_sci = np.full(shape, 8.)
    expected_weight = np.full(shape, 2.)
    expected_sci[:, 0, 2] = 10
    expected_weight[:, 0, 2] = 1
    for channel, row, col in [(2, 0, 1), (1, 1, 0), (2, 1, 0),
                              (3, 1, 0), (4, 1, 1), (0, 1, 2)]:
        expected_sci[channel, row, col] = np.nan
        expected_weight[channel, row, col] = 0

    output = tmp_path / "merged.fits"
    for _ in range(2):
        frame_merge(inputs, str(output), diagnostic_plot_filename="",
                    center_pixel_for_plots=[0, 0], target_radius=0,
                    load_frames_from_disk=True, mask_gap_union=True,
                    gap_region=[0, 4], target_detectors=["nrs1"],
                    nrs1_region=(0, 5), nrs2_region=(0, 5))
        with fits.open(output) as product:
            np.testing.assert_array_equal(product["SCI"].data, expected_sci)
            np.testing.assert_array_equal(product["WMAP"].data, expected_weight)
    assert np.load(tmp_path / "merged_gaps_v3.npy").shape == (3, 2, 2, 2)
