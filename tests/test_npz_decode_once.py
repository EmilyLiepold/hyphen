"""Exercise real saved pairs and NPZ member reuse, without FITS inputs."""
import json
from collections import Counter

import numpy as np
import pytest
from numpy.lib.npyio import NpzFile

from hyphen.dewiggler_stack import _load_fit_results, _save_fit_results


@pytest.mark.parametrize("count", [2, 4])
def test_saved_pair_decodes_each_member_once_and_retains_rows(tmp_path, monkeypatch, count):
    rows = []
    for i in range(count):
        fft = {} if i == count - 1 else {
            "freq": np.array([0.0, 0.5 + i]),
            "spectrum_fft": np.array([1 + 2j, i + 3j]),
            "residual_fft": np.array([3 - 1j, i + 4j]),
        }
        rows.append((np.arange(i + 2, dtype=float), {"order": i}, [i, count - i],
                     np.array([i, np.nan, i + 1], dtype=np.float64), fft,
                     float(i + 0.5), f"row {i}", i - 1))
    base = str(tmp_path / "fits.pkl.gz")
    _save_fit_results(rows, base)
    calls = Counter()
    original = NpzFile.__getitem__

    def observe(self, key):
        calls[key] += 1
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", observe)
    loaded = _load_fit_results(base)
    assert calls == {key: 1 for key in ("fft_freqs", "fft_spectra", "fft_residuals",
                                        "xsols", "target_indices", "dewiggled_spectra",
                                        "base_x2s", "target_slices")}
    assert len(loaded) == count
    for expected, actual in zip(rows, loaded, strict=True):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[0].dtype == expected[0].dtype
        assert actual[1] == expected[1] and actual[2] == expected[2]
        assert isinstance(actual[2], list)
        np.testing.assert_array_equal(actual[3], expected[3])
        assert actual[3].dtype == expected[3].dtype
        assert set(actual[4]) == {"freq", "spectrum_fft", "residual_fft"}
        for key in actual[4]:
            expected_fft = expected[4].get(key, np.array([]))
            np.testing.assert_array_equal(actual[4][key], expected_fft)
            assert actual[4][key].dtype == expected_fft.dtype
        assert isinstance(actual[5], float)
        assert isinstance(actual[7], np.integer)
        assert actual[5:] == expected[5:]


@pytest.mark.parametrize("count", [2, 4])
def test_spectra_rows_share_one_decoded_backing_array(tmp_path, count):
    rows = [(np.array([i]), {"order": i}, [i, i], np.array([i, i + 1.]), {},
             float(i), str(i), 0) for i in range(count)]
    base = str(tmp_path / "shared")
    _save_fit_results(rows, base)
    loaded = _load_fit_results(base)
    # Disjoint rows share the same single decoded spectra array as their base.
    assert all(row[3].base is loaded[0][3].base for row in loaded[1:])


def test_empty_pair_does_not_access_missing_members(tmp_path, monkeypatch):
    base = str(tmp_path / "empty")
    np.savez_compressed(base + ".npz", unrelated=np.array([1]))
    with open(base + ".json", "w") as handle:
        json.dump({"n": 0, "model_details": [], "log_strings": []}, handle)
    calls = []
    original = NpzFile.__getitem__

    def observe(self, key):
        calls.append(key)
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", observe)
    assert _load_fit_results(base) == []
    assert calls == []


def test_missing_first_fft_member_keeps_key_error(tmp_path):
    base = str(tmp_path / "missing")
    np.savez_compressed(base + ".npz", unrelated=np.array([1]))
    with open(base + ".json", "w") as handle:
        json.dump({"n": 1, "model_details": [{}], "log_strings": [""]}, handle)
    with pytest.raises(KeyError, match="fft_freqs"):
        _load_fit_results(base)


def test_short_first_fft_member_raises_before_later_missing_member(tmp_path):
    base = str(tmp_path / "short")
    np.savez_compressed(base + ".npz", fft_freqs=np.array([]))
    with open(base + ".json", "w") as handle:
        json.dump({"n": 1, "model_details": [{}], "log_strings": [""]}, handle)
    with pytest.raises(IndexError):
        _load_fit_results(base)
