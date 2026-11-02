"""Legacy mask YAML must not be silently ignored by merger dispatch."""

from unittest.mock import Mock

import pytest

from hyphen import frame_merger


def _write_config(path, *, generic="", merge="", extra=""):
    path.write_text(
        "step_type: generic\n"
        f"{generic}"
        "---\n"
        "step_type: merge\n"
        "label: test\n"
        "input_filename_prototype: input_FRAME_DETECTOR.fits\n"
        f"output_filename: {path.parent / 'merged.fits'}\n"
        "target_frames: [f1, f2]\n"
        "target_detectors: [nrs1, nrs2]\n"
        f"{merge}{extra}"
    )


@pytest.mark.parametrize("key", ["mask_filename", "mask_file"])
@pytest.mark.parametrize("layer", ["generic", "merge"])
@pytest.mark.parametrize("value", ["old-mask.json", "null"])
def test_legacy_options_rejected_before_merger_output(tmp_path, monkeypatch, key, layer, value):
    config = tmp_path / "config.yaml"
    _write_config(config, **{layer: f"{key}: {value}\n"})
    merger = Mock(side_effect=AssertionError("merger must not run"))
    monkeypatch.setattr(frame_merger, "frame_merge", merger)

    with pytest.raises(ValueError, match=rf"{key}.*deprecated.*mask_config"):
        frame_merger.frame_merge_wrapper(str(config))

    merger.assert_not_called()
    assert {p.name for p in tmp_path.iterdir()} == {"config.yaml"}


def test_step_null_does_not_hide_legacy_generic_option(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_config(config, generic="mask_file: old.json\n", merge="mask_file: null\n")
    merger = Mock(side_effect=AssertionError("merger must not run"))
    monkeypatch.setattr(frame_merger, "frame_merge", merger)
    with pytest.raises(ValueError, match=r"mask_file.*deprecated.*mask_config"):
        frame_merger.frame_merge_wrapper(str(config))
    merger.assert_not_called()
    assert {p.name for p in tmp_path.iterdir()} == {"config.yaml"}


def test_supported_mask_config_is_resolved_and_forwarded(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_config(config, extra="mask_config:\n  - frame: f2\n    detector: nrs1\n    pixels: [[2, 1]]\n")
    merger = Mock()
    monkeypatch.setattr(frame_merger, "frame_merge", merger)
    monkeypatch.setattr(frame_merger.logging, "basicConfig", lambda **kwargs: None)

    frame_merger.frame_merge_wrapper(str(config))

    merger.assert_called_once()
    assert merger.call_args.args[0] == [
        ["input_f1_nrs1.fits", "input_f1_nrs2.fits"],
        ["input_f2_nrs1.fits", "input_f2_nrs2.fits"],
    ]
    assert merger.call_args.kwargs["mask_config"] == [
        {"frame_idx": 1, "detector_idx": 0, "pixels": [[2, 1]]}
    ]
