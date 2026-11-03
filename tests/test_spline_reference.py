"""Finite median/spline reference checks; no science-fit tolerance is implied."""

import numpy as np
import pytest
from astropy.io import fits

from hyphen.dewiggle_lambda_space import _eval_pf, _pf_order
from hyphen.dewiggler_stack import InterpolatedPf, determine_reference_spectrum


ATOL = 1e-11  # Float64 interpolation allowance, not spectral-feature acceptance.


def reference(values, window, iterations=0):
    values = np.asarray(values, dtype=float)
    axis = np.linspace(1, 2, len(values))
    cube = values[:, None, None].copy()
    with fits.HDUList([
        fits.PrimaryHDU(), fits.ImageHDU(cube),
        fits.ImageHDU(np.ones_like(cube)),
        fits.ImageHDU(np.zeros(cube.shape, dtype=np.uint32)),
    ]) as hdus:
        raw, pf, smoothed = determine_reference_spectrum(
            "finite fixture", axis, hdus, np.zeros((1, 1)), 0, 0, -1,
            smooth_model_window=window, reference_spectra_percentile_cut=0,
            sigma_clipping_iterations=iterations, sigma_clipping_threshold=1)
    np.testing.assert_array_equal(raw, values)
    assert isinstance(pf, InterpolatedPf)
    assert pf.effective_order == _pf_order(pf) == window // 2
    assert len(pf) == window // 2 + 1
    np.testing.assert_allclose(pf(axis), smoothed, rtol=0, atol=ATOL)
    np.testing.assert_allclose(_eval_pf(pf, axis), smoothed, rtol=0, atol=ATOL)
    return axis, pf, smoothed


@pytest.mark.parametrize("window, expected", [
    (3, [1, 2, 8, 3, 7, 4, 6, 5, 5]),
    (5, [1, 2, 3, 7, 4, 6, 5, 5, 5]),
])
def test_nearest_edge_medians(window, expected):
    # Literal medians of each odd window, with the nearest endpoint repeated.
    _, _, smoothed = reference([1, 9, 2, 8, 3, 7, 4, 6, 5], window)
    np.testing.assert_array_equal(smoothed, expected)


@pytest.mark.parametrize("window", [1, 3, 5])
def test_affine_off_grid_evaluation_and_complexity(window):
    axis, pf, smoothed = reference([10, 11, 12, 13, 14], window)
    np.testing.assert_array_equal(smoothed, 4 * axis + 6)
    probes = np.array([1.125, 1.375, 1.625, 1.875])
    expected = 4 * probes + 6
    np.testing.assert_allclose(pf(probes), expected, rtol=0, atol=ATOL)
    np.testing.assert_allclose(_eval_pf(pf, probes), expected, rtol=0, atol=ATOL)


def test_quadratic_off_grid_evaluation():
    samples = [1, 1.5625, 2.25, 3.0625, 4]  # x**2 at the fixture's five nodes.
    _, pf, smoothed = reference(samples, 5)
    np.testing.assert_array_equal(smoothed, samples)  # Median preserves monotone samples.
    probes = np.array([1.125, 1.375, 1.625, 1.875])
    expected = [1.265625, 1.890625, 2.640625, 3.515625]  # x**2 between nodes.
    np.testing.assert_allclose(pf(probes), expected, rtol=0, atol=ATOL)
    np.testing.assert_allclose(_eval_pf(pf, probes), expected, rtol=0, atol=ATOL)


@pytest.mark.parametrize("iterations, expected", [
    (0, [0, 10, 10, 10, 20, 10, 10, 10, 30]),
    (1, [0, 10, 10, 10, 10, 10, 10, 10, 30]),
])
def test_clipping_replaces_deviations_before_refiltering(iterations, expected):
    # Seven strict-percentile samples: initial RMS = sqrt(300 / 7).
    # Threshold one rejects the three absolute residuals of ten.
    _, _, smoothed = reference([0, 10, 10, 20, 10, 20, 10, 10, 30], 3, iterations)
    np.testing.assert_array_equal(smoothed, expected)
