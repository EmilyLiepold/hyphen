"""The guide follows sqrt(nu), including reused artists, rather than fixed 5%.

Exercise the actual plotter with finite precomputed model samples, matching its
fitter-to-plotter interface. The canvas sink draws both figures without files.
No fitter, FFT estimator, threshold, scientific data or acceptance is tested.
"""
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from hyphen.dewiggle_lambda_space import plot_dewiggle_result


@pytest.mark.parametrize("reuse", [False, True])
def test_residual_guide_label_and_heights(reuse):
    x = np.linspace(0, 3, 8)
    lam = np.linspace(1, 2, 8)
    spectrum = 1 + 0.15 * np.sin(x)
    precomputed = dict(
        x_vals=x, y_for_fft=spectrum, med_px=x,
        good_indices=np.ones(8, dtype=bool), normalization=np.ones(8),
        normalization_constant=1.0,
        ww=lambda values, *args: np.ones_like(values),
        jwm=lambda *args: lambda values: 0.1 * np.sin(values),
        envelope=lambda values, *args: np.ones_like(values),
        background_prototype=lambda values: np.zeros_like(values),
        ps_continuum=np.ones(8), gal_continuum=np.zeros(8),
        spectrum_without_continuum=spectrum - 1,
        freq=np.array([0.0, 1.0, 2.0]),
        X_spec=np.array([1.0, 0.5, 0.25]),
        X_res=np.array([0.2, 0.1, 0.05]),
    )
    arguments = dict(
        trace_model=None, continuum_model=None, lam=lam,
        target_spectrum=spectrum, dq_spectrum=None,
        saved_xsol=[1, 0, 0.1, 0, 0, 1, 0.02, 0.01],
        saved_fit_params=dict(alpha_order=0, gamma_order=0, beta_order=0, order=1),
        plot=SimpleNamespace(savefig=lambda figure: figure.canvas.draw()),
        _precomputed=precomputed,
    )
    artists = plot_dewiggle_result(**arguments, saved_reduced_x2=0.04)
    first_guide = artists["res_hline_neg"]
    if reuse:
        artists = plot_dewiggle_result(
            **arguments, saved_reduced_x2=0.09, reusable_artists=artists
        )
        assert artists["res_hline_neg"] is first_guide
    height = 0.3 if reuse else 0.2
    for key, expected in [("neg", -height), ("zero", 0), ("pos", height)]:
        line = artists[f"res_hline_{key}"]
        np.testing.assert_allclose(line.get_ydata(), [expected, expected],
                                   rtol=0, atol=1e-14)
        assert line.get_linestyle() == "--"
    label = r"$\pm\sqrt{\nu}$ guide"
    assert first_guide.get_label() == label
    legend_labels = [text.get_text() for text in artists["axes2"][1].get_legend().texts]
    assert label in legend_labels
    assert "5% residual" not in legend_labels
