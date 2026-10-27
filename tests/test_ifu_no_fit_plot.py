"""A missing fit should not prevent its diagnostic from being prepared."""

from unittest.mock import Mock

import numpy as np
import pytest

from hyphen import dewiggler_stack as stack


DETAILS = {"alpha_order": 0, "beta_order": 0, "gamma_order": 0, "order": 0}


def _plot(monkeypatch, *, slices, preferred_slice, xsol):
    trace_factory = Mock(return_value=(lambda wave: wave, lambda wave: wave))
    map_plot = Mock()
    result_plot = Mock(return_value="artists")
    monkeypatch.setattr(stack, "build_spaxel_map_plot_ifu", map_plot)
    monkeypatch.setattr(stack, "plot_dewiggle_result", result_plot)
    coords = [(np.array([10. + i]), np.array([20. + i])) for i in range(len(slices))]
    row = (xsol, DETAILS, [0, 0], np.ones(4), {}, 1., "", preferred_slice)
    result = stack._plot_single_spaxel(
        x=row, xindex=0, diagnostic_fraction=1,
        _spaxel_world_coords=coords, slices_to_use=slices,
        trace_model_factory=trace_factory, spaxel_distance=np.zeros((1, 1)),
        axis_pixel=np.arange(1), axis_slice=np.arange(1), axis_spectral=np.arange(4.),
        image=np.ones((1, 1)), target_pixel=[0, 0], _det_image=None, bars=[],
        dewiggling_radius=.45, central_reference_spectrum_radius=.1,
        _sci_data=np.ones((4, 1, 1)), _dq_data=np.zeros((4, 1, 1)),
        _wgt_data=np.ones((4, 1, 1)), weight_based_masking_threshold=None,
        masked_regions=[], pf=np.array([1.]), pf_back=np.array([1.]),
        plot=None, reusable_artists=None,
    )
    return result, trace_factory, map_plot, result_plot


def test_no_fit_missing_sentinel_uses_first_available_trace(monkeypatch):
    result, trace, map_plot, plot = _plot(monkeypatch, slices=[1, 2], preferred_slice=0, xsol=None)
    assert result == "artists"
    trace.assert_called_once_with(10., 20.)
    map_plot.assert_called_once()
    np.testing.assert_array_equal(plot.call_args.kwargs["saved_xsol"], stack._make_zero_xsol(DETAILS))
    assert "[NO VALID FIT]" in plot.call_args.kwargs["plot_label"]


def test_no_fit_with_present_sentinel_keeps_selected_trace(monkeypatch):
    _, trace, _, plot = _plot(monkeypatch, slices=[1, 0], preferred_slice=0, xsol=None)
    trace.assert_called_once_with(11., 21.)
    assert "[NO VALID FIT]" in plot.call_args.kwargs["plot_label"]


def test_adopted_fit_keeps_its_selected_trace(monkeypatch):
    xsol = np.array([1., 0., 1., 0., 0., 1.])
    _, trace, _, plot = _plot(monkeypatch, slices=[1, 2], preferred_slice=2, xsol=xsol)
    trace.assert_called_once_with(11., 21.)
    np.testing.assert_array_equal(plot.call_args.kwargs["saved_xsol"], xsol)
    assert "[NO VALID FIT]" not in plot.call_args.kwargs["plot_label"]


def test_adopted_fit_with_unknown_slice_still_errors(monkeypatch):
    with pytest.raises(ValueError, match="0 is not in list"):
        _plot(monkeypatch, slices=[1], preferred_slice=0, xsol=np.ones(6))
