from unittest.mock import Mock

import numpy as np
import pytest

from hyphen.dewiggler_stack import determine_target_pixel_ifu


@pytest.mark.parametrize("kwargs,expected_xy,wcs_calls", [
    pytest.param(dict(target_pixel=[2, 3], target_radec=[12.34, -5.6]),
                 [2, 3], 0, id="explicit_list_overrides_RA"),
    pytest.param(dict(target_pixel=(2, 3), target_radec=(12.34, -5.6)),
                 [2, 3], 0, id="explicit_tuple_overrides_RA"),
    pytest.param({}, [4, 2], 0, id="omitted_center_default_tuple_RA"),
    pytest.param(dict(target_pixel=[-1, -1]), [4, 2], 0,
                 id="YAML_list_center_sentinel"),
    pytest.param(dict(target_pixel=np.array([2, 3])), [2, 3], 0,
                 id="NumPy_explicit_center"),
    pytest.param(dict(target_radec=[0.0, 0.0]), [4, 2], 0,
                 id="omitted_center_YAML_list_RA_sentinel"),
    pytest.param(dict(target_radec=(12.34, -5.6)), [3, 2], 1,
                 id="provided_RA_fractional_WCS"),
])
def test_ifu_centering_representations(kwargs, expected_xy, wcs_calls):
    image = np.zeros((3, 5), dtype=float)
    image[0, 0], image[1, 3] = np.nan, 12.0
    wcs = Mock()
    wcs.world_to_pixel.return_value = tuple(np.float64(x) for x in (1.6, 0.6, 0.0))
    actual = determine_target_pixel_ifu(image=image, frame_3d_wcs=wcs, **kwargs)
    np.testing.assert_array_equal(actual, expected_xy)
    assert actual.dtype.kind in "iu"
    assert wcs.world_to_pixel.call_count == wcs_calls
    if wcs_calls:
        coordinates, wavelength = wcs.world_to_pixel.call_args.args
        assert (coordinates.ra.deg, coordinates.dec.deg) == (12.34, -5.6)
        assert wavelength.to_value("micron") == 2.0
