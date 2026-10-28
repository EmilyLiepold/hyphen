from copy import deepcopy

import pytest

from hyphen import dewiggler_stack as stack
from hyphen.batch_dewiggle_ifu_step import batch_dewiggle_ifu_default_options
from hyphen.yaml_helpers import yaml_layerer


@pytest.mark.parametrize("plot", ["plots/PIXEL_FRAME_DETECTOR.pdf", False])
def test_two_centers_chain_3d_inputs_and_forward_options(monkeypatch, plot):
    options = {
        "input_2d_filename_prototype": "inputs/FRAME_DETECTOR_crf.fits",
        "input_3d_filename_prototype": "inputs/FRAME_DETECTOR_s3d.fits",
        "output_filename_prototype": "outputs/PIXEL_FRAME_DETECTOR.fits",
        "plot_filename_prototype": plot,
        "central_pixel": [[38, 33], [40, 35]],
        "target_frames": ["00001", "00002"],
        "n_jobs": 3,
        "dewiggling_radius": 0.45,
    }
    getter = yaml_layerer([batch_dewiggle_ifu_default_options, options])
    keys = batch_dewiggle_ifu_default_options.keys() | options.keys()
    overridden = {"central_pixel", "input_3d_filename_prototype",
                  "output_filename_prototype", "plot_filename_prototype"}
    expected = deepcopy({k: getter(k) for k in keys - overridden})
    calls = []

    def capture_single_pass(partial_getter):
        calls.append({key: partial_getter(key) for key in keys})
        assert partial_getter("absent_option", default="fallback") == "fallback"

    monkeypatch.setattr(stack, "parse_yaml_arguments_and_dewiggle_ifu", capture_single_pass)
    assert stack.parse_yaml_arguments_and_batch_dewiggle_ifu(getter) == 0
    assert len(calls) == 2
    assert [c["central_pixel"] for c in calls] == [[38, 33], [40, 35]]
    assert [c["input_3d_filename_prototype"] for c in calls] == [
        "inputs/FRAME_DETECTOR_s3d.fits", "outputs/38_33_FRAME_DETECTOR.fits"]
    assert [c["output_filename_prototype"] for c in calls] == [
        "outputs/38_33_FRAME_DETECTOR.fits", "outputs/40_35_FRAME_DETECTOR.fits"]
    expected_plots = ["plots/38_33_FRAME_DETECTOR.pdf", "plots/40_35_FRAME_DETECTOR.pdf"]
    assert [c["plot_filename_prototype"] for c in calls] == (expected_plots if plot else [False, False])
    for call in calls:
        assert {k: call[k] for k in keys - overridden} == expected
