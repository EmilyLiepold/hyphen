from functools import partial

import numpy as np
import pytest

from hyphen.dewiggle_lambda_space import build_wiggly_model, generic_envelope


@pytest.mark.parametrize("change", [
    "interiors", "mutation", "identical", "coordinate_dtype", "parameter_dtype",
])
def test_cached_components_match_fresh_callables(change):
    coordinates = np.array([-0.5, -0.25, 0.125, 0.5])
    parameters = np.array([
        0.85, 0.14, 0.35, -0.12, 1.0, 0.18, 1.0, 0.0, 1.03, 0.08, -0.04,
    ])

    def model():
        prediction, _, _, _, jacobian = build_wiggly_model(
            partial(generic_envelope, min_x=-0.5, max_x=0.5),
            lambda x: 0.6 + 0.04 * x,
            dict(alpha_order=1, gamma_order=1, beta_order=1, order=1),
            x_min=-0.5, x_max=0.5,
        )
        return prediction, jacobian

    cached = model()
    for function in cached:
        assert np.all(np.isfinite(function(coordinates, *parameters)))
    if change == "interiors":
        coordinates = np.array([-0.5, -0.1, 0.35, 0.5])
    elif change == "mutation":
        coordinates[1:3] = [-0.1, 0.35]
    elif change == "coordinate_dtype":
        coordinates = coordinates.view(np.int64)
    elif change == "parameter_dtype":
        parameters = parameters.view(np.int64)

    for reused, fresh in zip(cached, model()):
        actual = reused(coordinates, *parameters)
        expected = fresh(coordinates, *parameters)
        assert np.all(np.isfinite(actual))
        assert np.all(np.isfinite(expected))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
