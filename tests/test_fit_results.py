import json

import numpy as np

from hyphen.dewiggler_stack import _load_fit_results, _save_fit_results


def test_heterogeneous_ifu_fit_results_round_trip(tmp_path):
    orders = [
        dict(alpha_order=1, gamma_order=0, beta_order=1, order=1),
        dict(alpha_order=2, gamma_order=1, beta_order=2, order=2),
        dict(alpha_order=0, gamma_order=0, beta_order=0, order=0),
    ]
    ffts = [
        {"freq": np.array([0.0, 0.5]),
         "spectrum_fft": np.array([2 + 1j, -3j]),
         "residual_fft": np.array([0.1 - 0.2j, 0.3j])},
        {"freq": np.array([0.0, 0.25, 0.5]),
         "spectrum_fft": np.array([4j, 1 + 2j, -5]),
         "residual_fft": np.array([0.2j, 0.1, 0.4 - 0.3j])},
        {},
    ]
    rows = [
        (np.arange(10, dtype=float), orders[0], [4, 1],
         np.array([1.0, np.nan, 2.0, 3.0, 4.0]), ffts[0], 0.012,
         "spaxel (4,1): accepted\n", -1),
        (np.arange(15, dtype=float) / 2, orders[1], [1, 3],
         np.array([4.0, np.nan, 5.0, 6.0, 7.0]), ffts[1], 0.008,
         "spaxel (1,3): escalated\nsecond line\n", 1),
        (None, orders[2], [8, 2],
         np.array([7.0, np.nan, 8.0, 9.0, 10.0]), ffts[2], 1e10,
         "spaxel (8,2): no valid fit\n", 0),
    ]
    base = tmp_path / "cube.pkl.gz"
    _save_fit_results(rows, str(base))
    assert {p.name for p in tmp_path.iterdir()} == {
        "cube.pkl.gz.npz", "cube.pkl.gz.json"}
    metadata = json.loads((tmp_path / "cube.pkl.gz.json").read_text())
    assert metadata == dict(n=3, model_details=orders,
                            log_strings=[row[6] for row in rows])
    loaded = _load_fit_results(str(base))
    assert len(loaded) == len(rows)
    for expected, actual in zip(rows, loaded, strict=True):
        assert len(actual) == 8
        if expected[0] is None:
            assert actual[0] is None
        else:
            np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
        assert actual[2] == expected[2]
        np.testing.assert_array_equal(actual[3], expected[3])
        assert set(actual[4]) == {"freq", "spectrum_fft", "residual_fft"}
        for key in actual[4]:
            np.testing.assert_array_equal(actual[4][key], expected[4].get(key, []))
        assert actual[5:] == expected[5:]
