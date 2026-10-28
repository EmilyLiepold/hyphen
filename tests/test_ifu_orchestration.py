from unittest.mock import Mock, call, mock_open

import pytest

from hyphen import dewiggler_stack as stack
from hyphen.batch_dewiggle_ifu_step import batch_dewiggle_ifu_default_options
from hyphen.yaml_helpers import yaml_layerer


@pytest.mark.parametrize("plot", ["plots/PIXEL_FRAME_DETECTOR.pdf", False])
def test_batch_expands_and_chains_each_frame_detector(monkeypatch, plot):
    options = {
        "input_2d_filename_prototype": "inputs/FRAME_DETECTOR_crf.fits",
        "input_3d_filename_prototype": "inputs/FRAME_DETECTOR_s3d.fits",
        "output_filename_prototype": "outputs/PIXEL_FRAME_DETECTOR.fits",
        "plot_filename_prototype": plot,
        "central_pixel": [[38, 33], [40, 35]],
        "target_frames": ["00001", "00002"],
        "target_detectors": ["nrs1", "nrs2"],
        "dewiggling_radius": 0.45,
        "diagnostic_fraction": 3,
        "n_jobs": "1",
        "target_radec": [187.7, 12.4],
        "mask_bad_fits": False,
        "weight_based_masking_threshold": 0.6,
        "null_fit_improvement_threshold": 0.007,
        "target_wiggly_pixels": [[2, 3]],
    }
    fit, logs, configure = Mock(), mock_open(), Mock()
    monkeypatch.setattr(stack, "dewiggler_stack_ifu", fit)
    monkeypatch.setattr(stack, "open", logs, raising=False)
    monkeypatch.setattr(stack.logging, "basicConfig", configure)
    getter = yaml_layerer([batch_dewiggle_ifu_default_options, options])

    assert stack.parse_yaml_arguments_and_batch_dewiggle_ifu(getter) == 0
    expected = [
        ([38, 33], "inputs", "00001_nrs1"),
        ([38, 33], "inputs", "00001_nrs2"),
        ([38, 33], "inputs", "00002_nrs1"),
        ([38, 33], "inputs", "00002_nrs2"),
        ([40, 35], "outputs/38_33", "00001_nrs1"),
        ([40, 35], "outputs/38_33", "00001_nrs2"),
        ([40, 35], "outputs/38_33", "00002_nrs1"),
        ([40, 35], "outputs/38_33", "00002_nrs2"),
    ]
    assert len(fit.call_args_list) == len(logs.call_args_list) == len(configure.call_args_list) == 8
    outputs = []
    for actual, log_call, config_call, (center, previous, case) in zip(
            fit.call_args_list, logs.call_args_list, configure.call_args_list, expected):
        pixel = f"{center[0]}_{center[1]}"
        input_cube = f"inputs/{case}_s3d.fits" if previous == "inputs" else f"{previous}_{case}.fits"
        output = f"outputs/{pixel}_{case}.fits"
        assert actual.args == (f"inputs/{case}_crf.fits", input_cube, output)
        assert actual.kwargs["target_pixel"] == center
        assert actual.kwargs["plot"] == (f"plots/{pixel}_{case}.pdf" if plot else False)
        for key in ("dewiggling_radius", "diagnostic_fraction", "mask_bad_fits",
                    "weight_based_masking_threshold", "null_fit_improvement_threshold"):
            assert actual.kwargs[key] == options[key]
        assert actual.kwargs["n_jobs"] == 1
        assert actual.kwargs["target_radec"] == (187.7, 12.4)
        assert actual.kwargs["target_wiggle_pixels"] == [[2, 3]]
        logfile = f"outputs/{pixel}_{case}.log"
        assert log_call == call(logfile, "w")
        assert config_call.kwargs["filename"] == logfile
        outputs.append(output)
    assert len(set(outputs)) == 8
