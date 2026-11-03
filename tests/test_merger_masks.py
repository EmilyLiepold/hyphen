import numpy as np
import pytest

from hyphen.frame_merger import _apply_pixel_masks


@pytest.mark.parametrize(("selection", "expected_index"), [
    ({"frame_idx": 1, "detector_idx": 0}, (1, 0)),
    ({"detector_idx": 1}, (slice(None), 1)),
    ({"frame_idx": 0}, (0, slice(None))),
    ({}, (slice(None), slice(None))),
])
def test_mask_coordinate_and_selector_scope(selection, expected_index):
    initial = np.arange(96, dtype=float).reshape(2, 2, 3, 2, 4)
    actual = initial.copy()
    expected = initial.copy()
    expected[expected_index + (slice(None), 1, 3)] = np.nan

    _apply_pixel_masks(actual, [dict(selection, pixels=[[3, 1]])], "SCI")

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("entry", [
    {"frame_idx": -1, "pixels": [[0, 0]]},
    {"frame_idx": 2, "pixels": [[0, 0]]},
    {"detector_idx": -1, "pixels": [[0, 0]]},
    {"detector_idx": 2, "pixels": [[0, 0]]},
    {"pixels": [[-1, 0]]},
    {"pixels": [[0, -1]]},
    {"pixels": [[4, 1]]},
    {"pixels": [[0, 2]]},
    {"pixels": [[0, 0], [4, 1]]},
])
def test_invalid_entry_raises_before_mutating(entry):
    initial = np.arange(96, dtype=float).reshape(2, 2, 3, 2, 4)
    actual = initial.copy()

    with pytest.raises(ValueError):
        _apply_pixel_masks(actual, [entry], "SCI")

    np.testing.assert_array_equal(actual, initial)
