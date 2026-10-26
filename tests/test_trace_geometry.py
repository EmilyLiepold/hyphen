"""Nearest-trace identity on the sky, with deterministic synthetic WCS traces."""
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy import units as u

from hyphen.dewiggler_stack import get_2d_trace_model


@pytest.mark.parametrize(
    "query,candidates,expected_index",
    [
        ((20.0, 30.0), [(20.002, 30.001), (20.0001, 30.0002)], 1),
        ((359.999, 0.0), [(0.001, 0.0), (359.99, 0.0)], 0),
        ((0.0, 80.0), [(0.01, 80.0), (0.0, 80.005)], 0),
    ],
    ids=["ordinary", "ra-wrap", "high-declination"],
)
def test_trace_selection_matches_spherical_oracle(query, candidates, expected_index):
    """Check actual factory selection and both returned pixel-coordinate fits."""
    sky = np.asarray(candidates)
    separations = SkyCoord(*query, unit=u.deg).separation(
        SkyCoord(sky[:, 0], sky[:, 1], unit=u.deg)
    )
    oracle_index = int(np.argmin(separations))
    assert oracle_index == expected_index
    assert abs(np.diff(separations.arcsec)[0]) > 1.0  # No ambiguous nearest tie.

    def forward(x, y):
        # Ignore random detector positions: present the same two-trace library.
        assert len(x) == len(y) == 20
        return (
            np.resize(sky[:, 0], len(x)),
            np.resize(sky[:, 1], len(x)),
            np.linspace(1.0, 2.0, len(x)),
        )

    def inverse(ra, dec, lam):
        # Distinct, exactly quadratic detector traces identify the chosen sky row.
        np.testing.assert_array_equal(ra, np.repeat(sky[:, 0], 10))
        np.testing.assert_array_equal(dec, np.repeat(sky[:, 1], 10))
        np.testing.assert_allclose(lam, np.tile(np.linspace(1.0, 2.0, 10), 2),
                                   rtol=0, atol=1e-14)
        identity = np.repeat(np.arange(2), 10)
        return 10 * identity + lam + 0.1 * lam**2, -20 * identity + 2 * lam - 0.3 * lam**2

    wcs = SimpleNamespace(pixel_to_world_values=forward, world_to_pixel_values=inverse)
    image = SimpleNamespace(data=np.zeros((2, 2)))
    trace_y, trace_x = get_2d_trace_model(wcs, image, n_target_traces=2)(*query)
    lam = np.array([1.0, 1.25, 1.75, 2.0])
    # Tolerance is arithmetic reconstruction in synthetic pixels, not science accuracy.
    np.testing.assert_allclose(trace_x(lam), 10 * oracle_index + lam + 0.1 * lam**2,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(trace_y(lam), -20 * oracle_index + 2 * lam - 0.3 * lam**2,
                               rtol=0, atol=1e-8)
