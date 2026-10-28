from functools import partial

import numpy as np
import pytest

from hyphen.dewiggle_lambda_space import build_wiggly_model, generic_envelope


ORDERS = dict(alpha_order=2, gamma_order=2, beta_order=2, order=2)
X = np.array([-0.8, -0.58, -0.25, -0.02, 0.21, 0.52, 0.9])
PARAMETERS = np.array([
    0.85, 0.14, -0.09,  # alpha scale, P1, P2
    0.35, -0.12, 0.08,  # gamma scale, P1, P2
    1.0, 0.18, -0.11,  # beta scale, P1, P2
    1.0, 0.0, 1.03,  # cP, cG, frequency
    0.08, -0.04, 0.03, 0.02,  # sine/cosine amplitudes, harmonics 1 and 2
])
FREE_COLUMNS = [0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15]
FREE_NAMES = [
    "alpha0", "alpha1", "alpha2", "gamma0", "gamma1", "gamma2",
    "beta1", "beta2", "frequency", "a1", "b1", "a2", "b2",
]


def _background(x):
    return 0.6 + 0.04 * x + 0.03 * x**2


def _model(orders=ORDERS):
    envelope = partial(generic_envelope, min_x=-0.8, max_x=0.9)
    return build_wiggly_model(
        envelope, _background, orders, x_min=-0.8, x_max=0.9,
    )


def _expected(x):
    # Closed P1/P2 forms provide an oracle independent of the cached basis.
    t = 2 * (x + 0.8) / 1.7 - 1
    p2 = (3 * t**2 - 1) / 2
    alpha = 0.85 * (1 + 0.14 * t - 0.09 * p2)
    gamma = 0.35 * (1 - 0.12 * t + 0.08 * p2)
    beta = 1 + 0.18 * t - 0.11 * p2
    phase = 2 * np.pi * 1.03 * x
    wiggle = beta * (
        0.08 * np.sin(phase) - 0.04 * np.cos(phase)
        + 0.03 * np.sin(2 * phase) + 0.02 * np.cos(2 * phase)
    )
    return alpha * (1 + wiggle) + gamma * _background(x)


def test_model_matches_independent_legendre_and_harmonic_formula():
    model, *_ = _model()
    prediction = model(X, *PARAMETERS)
    assert prediction.shape == X.shape
    assert np.all(np.isfinite(prediction))
    np.testing.assert_allclose(prediction, _expected(X), rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("column", FREE_COLUMNS, ids=FREE_NAMES)
def test_free_parameter_jacobian_matches_central_difference(column):
    model, _, _, bounds, jacobian = _model()
    step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(PARAMETERS[column]))
    # beta0, cP and cG have near-fixed bounds and are deliberately excluded.
    assert bounds[0, column] < PARAMETERS[column] - step
    assert PARAMETERS[column] + step < bounds[1, column]
    plus, minus = PARAMETERS.copy(), PARAMETERS.copy()
    plus[column] += step
    minus[column] -= step
    finite_difference = (model(X, *plus) - model(X, *minus)) / (2 * step)
    analytic = jacobian(X, *PARAMETERS)
    assert analytic.shape == (len(X), len(PARAMETERS))
    assert np.all(np.isfinite(finite_difference))
    assert np.all(np.isfinite(analytic))
    np.testing.assert_allclose(
        analytic[:, column], finite_difference, rtol=2e-6, atol=2e-8,
    )


@pytest.mark.parametrize("envelope_order", [0, 2])
@pytest.mark.parametrize("block", [0, 1, 2], ids=["alpha", "gamma", "beta"])
def test_zero_scale_envelope_derivatives(envelope_order, block):
    coefficients = [PARAMETERS[i:i + envelope_order + 1].copy() for i in (0, 3, 6)]
    coefficients[block][0] = 0
    parameters = np.concatenate([*coefficients, PARAMETERS[9:]])
    orders = dict.fromkeys(("alpha_order", "gamma_order", "beta_order"), envelope_order)
    orders["order"] = 2
    _, _, _, _, jacobian = _model(orders)
    t = 2 * (X + 0.8) / 1.7 - 1
    basis = np.array([np.ones_like(X), t, (3 * t**2 - 1) / 2])[:envelope_order + 1]
    shapes = [np.concatenate(([1.0], c[1:])) @ basis for c in coefficients]
    alpha = coefficients[0][0] * shapes[0]
    beta = coefficients[2][0] * shapes[2]
    phase = 2 * np.pi * 1.03 * X
    harmonics = (
        0.08 * np.sin(phase) - 0.04 * np.cos(phase)
        + 0.03 * np.sin(2 * phase) + 0.02 * np.cos(2 * phase)
    )
    factors = (1 + beta * harmonics, _background(X), alpha * harmonics)
    expected = np.zeros((len(X), envelope_order + 1))
    expected[:, 0] = shapes[block] * factors[block]
    analytic = jacobian(X, *parameters)
    assert np.all(np.isfinite(analytic))
    start = block * (envelope_order + 1)
    np.testing.assert_allclose(
        analytic[:, start:start + envelope_order + 1], expected,
        rtol=2e-13, atol=2e-14,
    )
