"""IFU threshold defaults and YAML layering without running fits."""

import inspect
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from hyphen.batch_dewiggle_ifu_step import (
    batch_dewiggle_ifu_default_options as batch_defaults,
    batch_dewiggle_ifu_necessary_arguments as batch_required,
)
from hyphen.single_dewiggle_ifu_step import (
    single_dewiggle_ifu_default_options as single_defaults,
    single_dewiggle_ifu_necessary_arguments as single_required,
)
from hyphen.yaml_helpers import ingest_yaml


@pytest.mark.parametrize(
    "step_type,defaults,required,center",
    [
        ("dewiggle_ifu", single_defaults, single_required, "[38, 33]"),
        ("batch_dewiggle_ifu", batch_defaults, batch_required, "[[38, 33]]"),
    ],
    ids=["single", "batch"],
)
@pytest.mark.parametrize(
    "threshold_options,weight_threshold,null_threshold",
    [
        ("", 0.5, -0.1),
        ("weight_based_masking_threshold: null", None, -0.1),
        (
            "weight_based_masking_threshold: 0.8\nnull_fit_improvement_threshold: 0.02",
            0.8,
            0.02,
        ),
    ],
    ids=["defaults", "weight-null", "explicit-thresholds"],
)
def test_ifu_dispatch_resolves_valid_layers(
    step_type,
    defaults,
    required,
    center,
    threshold_options,
    weight_threshold,
    null_threshold,
):
    config = f"""step_type: generic
input_2d_filename_prototype: input_FRAME_DETECTOR_crf.fits
input_3d_filename_prototype: input_FRAME_DETECTOR_s3d.fits
output_filename_prototype: output_PIXEL_FRAME_DETECTOR.fits
target_frames: ['00001']
dewiggling_radius: 0.45
n_jobs: 2
---
step_type: {step_type}
label: mock_only
central_pixel: {center}
n_jobs: 3
{threshold_options}
"""
    target = Mock()
    descriptor = {
        "target_step_type": step_type,
        "target_function": target,
        "default_options": defaults,
        "necessary_arguments": required,
    }
    with patch("builtins.open", return_value=StringIO(config)) as opened:
        ingest_yaml("config.yaml", targets=[descriptor])
    opened.assert_called_once_with("config.yaml", "r")
    target.assert_called_once()
    getter = target.call_args.args[0]
    assert getter("dewiggling_radius") == 0.45
    assert getter("n_jobs") == 3
    assert getter("smooth_model_window") == 51
    assert {
        name: getter(name)
        for name in ("weight_based_masking_threshold", "null_fit_improvement_threshold")
    } == {
        "weight_based_masking_threshold": weight_threshold,
        "null_fit_improvement_threshold": null_threshold,
    }
    assert getter("n_jobs", override=[{"n_jobs": 4}]) == 4


def test_direct_ifu_threshold_defaults():
    from hyphen.dewiggler_stack import dewiggler_stack_ifu

    parameters = inspect.signature(dewiggler_stack_ifu).parameters
    assert parameters["weight_based_masking_threshold"].default == 0.5
    assert parameters["null_fit_improvement_threshold"].default == -0.1
