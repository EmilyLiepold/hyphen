"""Analytic reference-builder checks; no IFU fitting or real WCS is exercised."""

import numpy as np
import pytest
from astropy.io import fits

from hyphen.dewiggle_lambda_space import _eval_pf
from hyphen.dewiggler_stack import determine_reference_spectrum


ATOL = 1e-11  # Float64 arithmetic allowance, not a science-fit tolerance.


def cube_hdus(cube, dq=None):
    if dq is None:
        dq = np.zeros(cube.shape, dtype=np.uint32)
    return fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube),
                         fits.ImageHDU(np.ones(cube.shape)), fits.ImageHDU(dq)])


def constant_fit(raw, **options):
    axis = np.linspace(1, 2, len(raw))
    with cube_hdus(np.asarray(raw, dtype=float)[:, None, None]) as hdus:
        return determine_reference_spectrum(
            "fixture", axis, hdus, np.zeros((1, 1)), 0, 0, 0,
            reference_spectra_percentile_cut=0, **options)


def test_aperture_dq_nan_and_legendre_callable():
    axis = np.linspace(1, 2, 11)
    def polynomial(wavelength):
        x = 2 * wavelength - 3
        return 10 + 2 * x + 0.5 * (3 * x**2 - 1) / 2

    expected = polynomial(axis)
    cube = np.full((len(axis), 2, 3), 999.0)
    cube[:, 1, 0], cube[:, 1, 2] = expected - 1, expected + 1
    cube[4, 1, ::2] = expected[4], np.nan
    cube[5, 1, ::2] = np.nan
    dq = np.zeros(cube.shape, dtype=np.uint32)
    dq[:, 1, 1] = 2
    distances = np.array([[9, 0.2], [9, 0.3], [9, 0.4]])  # [x, y]
    with cube_hdus(cube, dq) as hdus:
        raw, pf, model = determine_reference_spectrum(
            "fixture", axis, hdus, distances, 0.2, 0.4, 2,
            reference_spectra_percentile_cut=0, sigma_clipping_iterations=0)
    expected_raw = expected.copy()
    expected_raw[5] = np.nan
    np.testing.assert_allclose(raw, expected_raw, rtol=0, atol=ATOL)
    np.testing.assert_allclose(model, expected, rtol=0, atol=ATOL)
    probes = np.array([1.125, 1.375, 1.875])
    assert callable(pf)
    np.testing.assert_allclose(_eval_pf(pf, probes), polynomial(probes),
                               rtol=0, atol=ATOL)


def test_percentile_boundaries_are_excluded():
    values = [0, 1, 2, 3, 4, 5, 7, 9, 100]
    axis = np.linspace(1, 2, len(values))
    with cube_hdus(np.asarray(values, dtype=float)[:, None, None]) as hdus:
        raw, pf, model = determine_reference_spectrum(
            "fixture", axis, hdus, np.zeros((1, 1)), 0, 0, 0,
            reference_spectra_percentile_cut=25, sigma_clipping_iterations=0)
    np.testing.assert_array_equal(raw, values)
    np.testing.assert_allclose(model, 4, rtol=0, atol=ATOL)
    np.testing.assert_allclose(_eval_pf(pf, axis), 4, rtol=0, atol=ATOL)


@pytest.mark.parametrize("iterations, expected", [(0, 10 + 4 / 15), (1, 10)])
def test_clipping_refits_after_rejecting_isolated_outlier(iterations, expected):
    values = [0] + [10] * 14 + [14, 100]
    raw, pf, model = constant_fit(
        values, sigma_clipping_iterations=iterations, sigma_clipping_threshold=2)
    np.testing.assert_array_equal(raw, values)
    np.testing.assert_allclose(model, expected, rtol=0, atol=ATOL)
    np.testing.assert_allclose(_eval_pf(pf, np.array([1.2, 1.8])), expected,
                               rtol=0, atol=ATOL)


def test_constant_reference_has_no_strict_percentile_samples():
    raw, pf, model = constant_fit([10] * 5)
    np.testing.assert_array_equal(raw, 10)
    np.testing.assert_array_equal(np.asarray(pf), [np.nan])
    np.testing.assert_array_equal(model, np.full(5, np.nan))
