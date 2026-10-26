import logging

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.optimize import curve_fit
from typing import Callable, Union
from functools import partial
logger = logging.getLogger(__name__)


def _eval_pf(
    pf: Union[np.ndarray, Callable, None],
    x: np.ndarray,
) -> np.ndarray:
    if pf is None:
        return np.zeros_like(x, dtype=float)
    if callable(pf):
        return np.asarray(pf(x), dtype=float)
    return np.polyval(pf, x)


def _pf_order(pf: Union[np.ndarray, Callable, None]) -> int:
    effective_order = getattr(pf, "effective_order", None)
    if effective_order is not None:
        return int(effective_order)
    degree_method = getattr(pf, "degree", None)
    if degree_method is not None and callable(degree_method):
        return int(degree_method())
    if pf is None or callable(pf):
        return 10
    return len(pf) - 1


def generic_envelope(x: np.typing.NDArray[np.float64],
                     *leg_coefficients: float,
                     min_x: float,
                     max_x: float) -> np.typing.NDArray[np.float64]:
    """
    Calculate the envelope function using Legendre polynomials.

    :param x: The input value for which the envelope function is
        calculated.
    :param *leg_coefficients: The coefficients of the Legendre polynomial which
        describes the envelope
    :param min_x: The minimum value of the x range to use for recaling the x values.
    :param max_x: The maximum value of the x range to use for recaling the x values.
    :return: The calculated value of the envelope function.
    """
    mod_leg_coefficients = [1.0] + list(leg_coefficients[1:])
    legendre = np.polynomial.Legendre(mod_leg_coefficients)
    rescaled_x = 2 * (x - min_x) / (max_x - min_x) - 1
    leg_values = leg_coefficients[0] * legendre(rescaled_x)

    return leg_values


def single_wiggle(
    x: np.typing.NDArray[np.float64],
    order: int,
    a: float,
    b: float,
    f: float
) -> np.typing.NDArray[np.float64]:
    """

    :param x: This is the x value at which the wiggle is evaluated
    :param order: This is the order of the wiggle (e.g., 1 for a first order
        wiggle, 2 for a second order)
    :param a: This is the amplitude of the sin term
    :param b: This is the amplitude of the cos term
    :param f: This is a fudge factor which alters the frequency of the wiggle.
        Typical value should be 1.0
    :return: The wiggle, evaluated at all points in x
    """
    frequency = 2 * order * np.pi * f
    sin_term = np.sin(frequency * x)
    cos_term = np.cos(frequency * x)

    return a * sin_term + b * cos_term


def build_wiggle_term(
    envelope_prototype: Callable[
        [np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]
    ],
    envelope_args: list,
    wiggle_args: list,
    f: float = 1,
) -> Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]:
    """

    :param envelope_prototype: This function defines a generic envelope. It
        should accept a list of x values as its first argument, then a list of
        parameters which define the envelope
    :param envelope_args: This is the list of parameters which define the
        envelope
    :param wiggle_args: This is a list of arguments which define the wiggle.
        These should pairs of sin and cos
    amplitudes for each order of the wiggle.
    :param f: This is the frequency fudge factor. Nominal value should be 1.0
    :return: A callable function which will return the wiggle model evaluated
        at the input x values
    """

    wiggle_args = list(wiggle_args)
    order = len(wiggle_args) // 2

    assert len(wiggle_args) % 2 == 0, \
        "The number of wiggle arguments must be even"

    assert (
        len(wiggle_args) == 2 * order
    ), "The number of wiggle arguments is not two per order..."

    def wiggle_term(
        x: np.typing.NDArray[np.float64],
        envelope_only: bool = False
    ) -> np.typing.NDArray[np.float64]:
        """
        Calculate the wiggle term for the given x values.

        This function computes the wiggle term by combining the sinusoidal part
        and the envelope function. The sinusoidal part is calculated by summing
        the contributions of multiple single wiggles, each defined by sine and
        cosine terms with specific amplitudes and frequencies. The envelope
        function is then applied to the sinusoidal part to obtain the final
        wiggle term.

        :param x: A list of values where the wiggle will be evaluated.
        :param envelope_only: A flag specifying whether to return the envelope
        or the full model.
        :return: The wiggle model evaluated at the input x values (or the envelope if
        envelope_only is True).
        """
        sinusoidal_part = np.zeros_like(x)

        # Loop through the orders of the wiggle, adding the single wiggle
        # associated with each
        for i in range(order):
            sinusoidal_part += single_wiggle(
                x, i + 1, wiggle_args[2 * i], wiggle_args[2 * i + 1], f
            )

        # Evaluate the envelope
        envelope = envelope_prototype(x, *envelope_args)

        # Construct and return the model
        return envelope * sinusoidal_part if not envelope_only else envelope

    return wiggle_term


def parse_wiggle_args(
    args: np.typing.NDArray[np.float64] | list[float],
    fit_orders: dict[str, int]
        ) -> tuple[
        list[float], list[float], float
        ] | tuple[
        np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], float
        ]:
    """
    Parses the wiggle arguments from the provided list of arguments.

    Args:
        args (list or np.ndarray): The list of arguments to parse.
        fit_orders (dict): A dictionary containing the fit orders for
            alpha, beta, gamma, and the wiggle.
    Returns:
        tuple: A tuple containing:
            - envelope_args (list or np.ndarray): The arguments related to the envelope.
            - wiggle_args (list or np.ndarray): The arguments related to the wiggle.
            - f (float): The frequency argument.
    """

    alpha_order = fit_orders["alpha_order"]
    beta_order = fit_orders["beta_order"]
    gamma_order = fit_orders["gamma_order"]
    order = fit_orders["order"]

    envelope_starting_index = alpha_order + gamma_order + 2
    envelope_ending_index = alpha_order + beta_order + gamma_order + 3

    envelope_args = args[envelope_starting_index: envelope_ending_index]

    wiggle_starting_index = alpha_order + beta_order + gamma_order + 6
    wiggle_ending_index = (
        alpha_order + beta_order + gamma_order + 6 + 2 * order
    )

    wiggle_args = args[wiggle_starting_index: wiggle_ending_index]

    frequency_index = alpha_order + beta_order + gamma_order + 5

    f = args[frequency_index]

    if isinstance(args, np.ndarray):
        return np.asarray(envelope_args), np.asarray(wiggle_args), float(f)
    else:
        return list(envelope_args), list(wiggle_args), float(f)


def plot_dewiggle_result(
    trace_model,
    continuum_model,
    lam,
    target_spectrum,
    dq_spectrum,
    saved_xsol,
    saved_fit_params,
    saved_reduced_x2,
    plot,
    plot_label="",
    background_pf=None,
    target_maskarray=None,
    reusable_artists=None,
    _precomputed=None,
):
    if _precomputed is not None:
        x_vals = _precomputed["x_vals"]
        y_for_fft = _precomputed["y_for_fft"]
        med_px = _precomputed["med_px"]
        good_indices = _precomputed["good_indices"]
        normalization = _precomputed["normalization"]
        normalization_constant = _precomputed["normalization_constant"]
        ww = _precomputed["ww"]
        jwm = _precomputed["jwm"]
        envelope = _precomputed["envelope"]
        background_prototype = _precomputed["background_prototype"]
        ps_continuum = _precomputed["ps_continuum"]
        gal_continuum = _precomputed["gal_continuum"]
        spectrum_without_continuum = _precomputed["spectrum_without_continuum"]
        freq = _precomputed["freq"]
        X_spec = _precomputed["X_spec"]
        X_res = _precomputed["X_res"]
        xsol = np.asarray(saved_xsol, dtype=float)
        fit_params = saved_fit_params
        reduced_x2 = saved_reduced_x2
    else:
        normalization = _eval_pf(continuum_model, lam)
        normalized_spectrum = target_spectrum / normalization
        good_indices = np.isfinite(normalized_spectrum)
        normalization_constant = np.nanmedian(normalized_spectrum[good_indices])
        normalized_spectrum = normalized_spectrum / normalization_constant

        good_indices = np.isfinite(normalized_spectrum)
        if dq_spectrum is None:
            dq_spectrum = np.zeros_like(target_spectrum)
        good_indices = (
            good_indices
            * (dq_spectrum[:] == 0)
            * (
                normalized_spectrum
                < np.nanpercentile(normalized_spectrum[good_indices], 99)
            )
            * (
                normalized_spectrum
                > np.nanpercentile(normalized_spectrum[good_indices], 1)
            )
        )
        if target_maskarray is not None:
            good_indices = good_indices * target_maskarray

        med_px = trace_model[0][0](lam)
        med_px -= np.nanmedian(med_px)
        inverse_model = partial(
            np.interp,
            xp=med_px[np.argsort(med_px)],
            fp=lam[np.argsort(med_px)]
        )

        background_normalization = np.nanmedian(
            _eval_pf(background_pf, lam) / _eval_pf(continuum_model, lam)
        )

        def background_prototype_exact(x):
            if background_normalization == 0:
                return np.zeros_like(x)
            else:
                return (
                    _eval_pf(background_pf, inverse_model(x))
                    / _eval_pf(continuum_model, inverse_model(x))
                    / background_normalization
                )

        background_prototype = background_prototype_exact

        x_vals = med_px[good_indices]
        y_for_fft = normalized_spectrum[good_indices]
        envelope = partial(generic_envelope, min_x=np.min(x_vals), max_x=np.max(x_vals))

        ww, _, jwm, _, _ = build_wiggly_model(
            envelope,
            background_prototype,
            fit_orders=saved_fit_params,
            x_min=float(np.min(x_vals)),
            x_max=float(np.max(x_vals)),
        )

        noise_spectrum = (
            np.ones_like(target_spectrum) * normalization * normalization_constant
        )
        normalized_noise_spectrum = (
            noise_spectrum / normalization / normalization_constant
        )
        noise_for_fft = normalized_noise_spectrum[good_indices]

        xsol = saved_xsol
        fit_params = saved_fit_params
        reduced_x2 = saved_reduced_x2

        ps_continuum = envelope(x_vals, *xsol[0 : fit_params["alpha_order"] + 1])
        gal_continuum = background_prototype(x_vals) * envelope(
            x_vals,
            *xsol[
                fit_params["alpha_order"]
                + 1 : fit_params["alpha_order"]
                + fit_params["gamma_order"]
                + 2
            ],
        )
        spectrum_without_continuum = y_for_fft - ps_continuum - gal_continuum
        residual_without_continuum = (
            y_for_fft - ps_continuum - gal_continuum - jwm(*xsol)(x_vals)
        )

        _fft_freq_step = 0.01
        _fft_period = 1.0 / _fft_freq_step
        _span = np.max(x_vals) - np.min(x_vals)
        _dt_nominal = _span / (100 * len(x_vals) - 1)
        _n_per_period = max(1, round(_fft_period / _dt_nominal))
        _n_fft = int(np.ceil(100 * len(x_vals) / _n_per_period)) * _n_per_period
        _dt_fft = _fft_period / _n_per_period
        fine_x_grid = np.linspace(np.min(x_vals), np.min(x_vals) + _n_fft * _dt_fft, _n_fft + 1)

        fine_grid_y_spec = np.interp(
            fine_x_grid,
            x_vals[np.argsort(x_vals)],
            spectrum_without_continuum[np.argsort(x_vals)]
        )
        fine_grid_y_res = np.interp(
            fine_x_grid,
            x_vals[np.argsort(x_vals)],
            residual_without_continuum[np.argsort(x_vals)]
        )
        X_spec = fft(fine_grid_y_spec)
        X_res = fft(fine_grid_y_res)
        n = len(X_spec)
        dt = fine_x_grid[1] - fine_x_grid[0]
        freq = np.fft.fftfreq(n, d=dt)
        freq_trim = 5
        freq_we_care_about = (freq[:] > -freq_trim) & (freq[:] < freq_trim)
        freq = freq[freq_we_care_about]
        X_spec = X_spec[freq_we_care_about]
        X_res = X_res[freq_we_care_about]

    logger.debug("Beginning plots")
    envelope_args, _, _ = parse_wiggle_args(
        xsol,
        fit_params,
    )
    envelope_args_str = ", ".join([f"{x:.2f}" for x in envelope_args])

    freq_trim = 5
    lambda_fullmodel_y = normalization_constant * normalization * ww(med_px, *xsol)
    lambda_wiggle_y = normalization_constant * normalization * jwm(*xsol)(med_px)
    lambda_dewiggled_y = target_spectrum - lambda_wiggle_y

    det_fullmodel_y = ww(x_vals, *xsol)
    det_wiggle_y = jwm(*xsol)(x_vals)
    det_dewiggled_y = y_for_fft - det_wiggle_y

    res_dewiggled_y = y_for_fft - det_fullmodel_y
    res_input_y = spectrum_without_continuum
    min_y = np.nanmin([np.nanmin(res_dewiggled_y), np.nanmin(res_input_y)])
    max_y = np.nanmax([np.nanmax(res_dewiggled_y), np.nanmax(res_input_y)])
    ax_ylim = np.nanmax(np.abs([min_y, max_y]))

    spec_fft_norm = np.abs(X_spec) / np.nanmax(np.abs(X_spec))
    res_fft_norm = np.abs(X_res) / np.nanmax(np.abs(X_spec))

    lambda_space_xlim = (np.min(lam[good_indices]), np.max(lam[good_indices]))
    x_space_xlim = (np.max(x_vals), np.min(x_vals))
    low_x = int(np.floor(x_space_xlim[0]))
    high_x = int(np.ceil(x_space_xlim[1]))

    should_close = reusable_artists is None

    if reusable_artists is None:
        fig, axes = plt.subplots(1, 1, figsize=(6.0, 3.75))
        fig2, axes2 = plt.subplots(3, 1, figsize=(6.0, 9.7))

        fig.subplots_adjust(left=0.1, right=0.98, top=0.87, bottom=0.12, hspace=0.0)
        fig2.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.06, hspace=0.1)

        axes.set_title(
            f"{plot_label}\n"
            f"RMS residual = {100 * reduced_x2**0.5:.2f}%"
        )
        axes2[0].set_title(
            f"{plot_label}\n"
            f"RMS residual = {100 * reduced_x2**0.5:.2f}%"
        )

        # axes[0].set_ylabel("Flux, arbitrary")
        # axes[0].plot(lam, target_spectrum, label="Input Spectrum", c="k")

        # yrange = get_ybounds([1e50, 1e-50], target_spectrum)
        # axes[0].plot(
        #     lam,
        #     target_spectrum
        #     - normalization_constant * normalization * jwm(*xsol)(med_px),
        #     label="Dewiggled Spectrum",
        #     c="r",
        # )

        # yrange = get_ybounds(
        #     yrange,
        #     target_spectrum
        #     - normalization_constant * normalization * jwm(*xsol)(med_px),
        # )

        # axes[0].set_ylim(0.9 * yrange[0], 1.1 * yrange[1])
        # axes[0].legend(fontsize=6)

        lam_input, = axes.plot(
            lam,
            target_spectrum,
            label="Input Spectrum",
            c="k",
            linewidth=0.5,
            
        )
        lam_fullmodel, = axes.plot(
            lam,
            lambda_fullmodel_y,
            label="Full Model",
            c="g",
            linewidth=0.5,
            
        )
        lam_wiggle, = axes.plot(
            lam,
            lambda_wiggle_y,
            label="Wiggle Model",
            c="b",
            linewidth=0.5,
            
        )
        lam_dewiggled, = axes.plot(
            lam,
            lambda_dewiggled_y,
            label="Dewiggled Spectrum",
            c="r",
            linewidth=0.5,
            
        )

        det_input, = axes2[0].plot(
            x_vals,
            y_for_fft,
            label="Input Spectrum",
            c="k",
            linewidth=0.5,
            
        )
        det_fullmodel, = axes2[0].plot(
            x_vals,
            det_fullmodel_y,
            label="Full Model",
            c="g",
            linewidth=0.5,
            
        )
        det_ps_cont, = axes2[0].plot(
            x_vals,
            ps_continuum,
            label=f"Central component: alpha={xsol[0]:.2f}",
            c="0.5",
            linestyle=":",
            linewidth=0.5,
            
        )
        gamma_0_idx = saved_fit_params["alpha_order"] + 1
        det_bg_cont, = axes2[0].plot(
            x_vals,
            gal_continuum,
            label=f"Background component: gamma={xsol[gamma_0_idx]:.2f}",
            c="0.5",
            linestyle="dashed",
            linewidth=0.5,
        )
        det_wiggle, = axes2[0].plot(
            x_vals,
            det_wiggle_y,
            linewidth=0.5,
            label=f"Wiggle Model\nl=({envelope_args_str})",
            c="b",
            
        )

        # axes2[0].plot(
        #     x_vals,
        #     jwm(*xsol)(x_vals, envelope_only=True),
        #     label=f"Wiggle Envelope,  l=({envelope_args_str})",
        #     c="b",
        #     linestyle=":",
        # )
        # axes2[0].plot(
        #     x_vals, -jwm(*xsol)(x_vals, envelope_only=True), c="b", linestyle=":"
        # )
        det_dewiggled, = axes2[0].plot(
            x_vals,
            det_dewiggled_y,
            label="Dewiggled Spectrum",
            c="r",
            linewidth=0.5,
            
        )

        handles, labels = axes2[0].get_legend_handles_labels()
        order = [0, 1, 4, 5, 2, 3]
        axes2[0].legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            fontsize=6,
            loc='lower center',
            framealpha=1.0,
        )

        axes2[0].set_ylabel("Flux, normalized (arbitrary)")
        axes2[0].set_xlabel('Detector pixel (arbitrary offset)')
        axes2[0].set_ylim(-0.3, 1.5)

        res_dewiggled, = axes2[1].plot(
            x_vals,
            res_dewiggled_y,
            label="Dewiggled spectrum minus continuum",
            c="r",
            linewidth=0.5,
            
        )
        res_input, = axes2[1].plot(
            x_vals,
            res_input_y,
            label="Input spectrum minus continuum",
            c="k",
            linewidth=0.5,
            
        )
        axes2[1].set_ylim(-ax_ylim, ax_ylim)

        axes2[1].set_ylabel("Residual")
        axes2[1].set_xlabel('Detector pixel (arbitrary offset)')
        res_hline_neg = axes2[1].axhline(
            -(reduced_x2**0.5),
            c="0.5",
            linestyle="--",
            label='5% residual',
            zorder=-10,
        )
        res_hline_zero = axes2[1].axhline(0, c="0.5", linestyle="--", zorder=-10)
        res_hline_pos = axes2[1].axhline(
            reduced_x2**0.5,
            c="0.5",
            linestyle="--",
            zorder=-10,
        )
        axes2[1].legend(fontsize=6, loc='lower center')

        fft_spec_stem = axes2[2].stem(freq, spec_fft_norm, "k")
        fft_res_stem = axes2[2].stem(freq, res_fft_norm, "r")
        axes2[2].scatter([], [], c='k', label='Input spectrum minus continuum')
        axes2[2].scatter([], [], c='r', label='Dewiggled spectrum minus continuum')

        axes2[2].set_xlim(0, freq_trim)
        axes2[2].set_ylim(0, 1.0)

        axes2[2].set_ylabel('Fourier spectrum')
        axes2[2].set_xlabel('Frequency (1 / detector pixels)')

        axes2[2].legend(fontsize=6, loc='upper right')
        axes.set_ylabel("Flux, arbitrary")
        axes.legend(fontsize=6, loc='lower center', framealpha=1.0)
        axes.set_xlabel("Wavelength (micron)")
        try:
            axes.set_ylim(
                np.nanmin((lambda_wiggle_y)[good_indices]),
                np.nanmax(target_spectrum[good_indices]),
            )
        except Exception:
            axes.set_ylim(0, np.nanmax(target_spectrum[good_indices]))

        axes.set_xlim(lambda_space_xlim)

        axes2[0].set_xlim(x_space_xlim)
        axes2[1].set_xlim(x_space_xlim)

        axes2[0].set_xticks(np.arange(low_x, high_x, 1))
        axes2[0].grid()
        axes2[1].set_xticks(np.arange(low_x, high_x, 1))
        axes2[1].grid(axis='x')

        reusable_artists = {
            'fig': fig,
            'fig2': fig2,
            'axes': axes,
            'axes2': axes2,
            'lam_input': lam_input,
            'lam_fullmodel': lam_fullmodel,
            'lam_wiggle': lam_wiggle,
            'lam_dewiggled': lam_dewiggled,
            'det_input': det_input,
            'det_fullmodel': det_fullmodel,
            'det_ps_cont': det_ps_cont,
            'det_bg_cont': det_bg_cont,
            'det_wiggle': det_wiggle,
            'det_dewiggled': det_dewiggled,
            'res_dewiggled': res_dewiggled,
            'res_input': res_input,
            'res_hline_pos': res_hline_pos,
            'res_hline_zero': res_hline_zero,
            'res_hline_neg': res_hline_neg,
            'fft_spec_stem': fft_spec_stem,
            'fft_res_stem': fft_res_stem,
        }
    else:
        should_close = False
        fig = reusable_artists['fig']
        fig2 = reusable_artists['fig2']
        axes = reusable_artists['axes']
        axes2 = reusable_artists['axes2']

        reusable_artists['lam_input'].set_xdata(lam)
        reusable_artists['lam_input'].set_ydata(target_spectrum)
        reusable_artists['lam_fullmodel'].set_xdata(lam)
        reusable_artists['lam_fullmodel'].set_ydata(lambda_fullmodel_y)
        reusable_artists['lam_wiggle'].set_xdata(lam)
        reusable_artists['lam_wiggle'].set_ydata(lambda_wiggle_y)
        reusable_artists['lam_dewiggled'].set_xdata(lam)
        reusable_artists['lam_dewiggled'].set_ydata(lambda_dewiggled_y)

        reusable_artists['det_input'].set_xdata(x_vals)
        reusable_artists['det_input'].set_ydata(y_for_fft)
        reusable_artists['det_fullmodel'].set_xdata(x_vals)
        reusable_artists['det_fullmodel'].set_ydata(det_fullmodel_y)
        reusable_artists['det_ps_cont'].set_xdata(x_vals)
        reusable_artists['det_ps_cont'].set_ydata(ps_continuum)
        reusable_artists['det_bg_cont'].set_xdata(x_vals)
        reusable_artists['det_bg_cont'].set_ydata(gal_continuum)
        reusable_artists['det_wiggle'].set_xdata(x_vals)
        reusable_artists['det_wiggle'].set_ydata(det_wiggle_y)
        reusable_artists['det_dewiggled'].set_xdata(x_vals)
        reusable_artists['det_dewiggled'].set_ydata(det_dewiggled_y)

        reusable_artists['res_dewiggled'].set_xdata(x_vals)
        reusable_artists['res_dewiggled'].set_ydata(res_dewiggled_y)
        reusable_artists['res_input'].set_xdata(x_vals)
        reusable_artists['res_input'].set_ydata(res_input_y)

        markerline, stemlines, baseline = reusable_artists['fft_spec_stem']
        markerline.set_data(freq, spec_fft_norm)
        stemlines.set_segments(
            [[[freq[j], 0], [freq[j], spec_fft_norm[j]]] for j in range(len(freq))]
        )
        markerline, stemlines, baseline = reusable_artists['fft_res_stem']
        markerline.set_data(freq, res_fft_norm)
        stemlines.set_segments(
            [[[freq[j], 0], [freq[j], res_fft_norm[j]]] for j in range(len(freq))]
        )

        reusable_artists['det_ps_cont'].set_label(
            f"Central component: alpha={xsol[0]:.2f}"
        )
        gamma_0_idx = saved_fit_params["alpha_order"] + 1
        reusable_artists['det_bg_cont'].set_label(
            f"Background component: gamma={xsol[gamma_0_idx]:.2f}"
        )
        reusable_artists['det_wiggle'].set_label(
            f"Wiggle Model\nl=({envelope_args_str})"
        )
        handles, labels = axes2[0].get_legend_handles_labels()
        axes2[0].legend(
            [handles[i] for i in [0, 1, 4, 5, 2, 3]],
            [labels[i] for i in [0, 1, 4, 5, 2, 3]],
            fontsize=6,
            loc='lower center',
            framealpha=1.0,
        )

        axes.set_title(
            f"{plot_label}\n"
            f"RMS residual = {100 * reduced_x2**0.5:.2f}%"
        )
        axes2[0].set_title(
            f"{plot_label}\n"
            f"RMS residual = {100 * reduced_x2**0.5:.2f}%"
        )

        reusable_artists['res_hline_neg'].set_ydata([-(reduced_x2**0.5), -(reduced_x2**0.5)])
        reusable_artists['res_hline_zero'].set_ydata([0, 0])
        reusable_artists['res_hline_pos'].set_ydata([reduced_x2**0.5, reduced_x2**0.5])

        try:
            axes.set_ylim(
                np.nanmin((lambda_wiggle_y)[good_indices]),
                np.nanmax(target_spectrum[good_indices]),
            )
        except Exception:
            axes.set_ylim(0, np.nanmax(target_spectrum[good_indices]))

        axes.set_xlim(lambda_space_xlim)
        axes2[0].set_xlim(x_space_xlim)
        axes2[1].set_xlim(x_space_xlim)
        axes2[1].set_ylim(-ax_ylim, ax_ylim)

        axes2[0].set_xticks(np.arange(low_x, high_x, 1))
        axes2[1].set_xticks(np.arange(low_x, high_x, 1))
        axes2[2].set_xlim(0, freq_trim)
        axes2[2].set_ylim(0, 1.0)

    plot.savefig(fig)
    plot.savefig(fig2)

    if should_close:
        plt.close(fig)
        plt.close(fig2)

    return reusable_artists


def convert_wiggle_solution(

    sol: np.typing.NDArray[np.float64],
    prior_fit_orders: dict[str, int],
    target_fit_orders: dict[str, int],
    padding_length: int = 100,
):
    """
    Converts a wiggle solution from one set of orders to another,
    with appropriate padding.

    Parameters:
    sol (np.typing.NDArray[np.float64]): The solution array containing prior solution
    prior_fit_orders (dict[str, int]): A dictionary containing the fit orders
        corresponding to the prior solution.
    target_fit_orders (dict[str,int]): A dictionary containing the fit orders
        which the solution should be converted to.
    padding_length (int, optional): The length of padding to apply to the
        parameters before conversion. Default is 100.

    Returns:
    np.ndarray: The converted solution array with the target orders and padding applied.
    """

    prior_alpha_order = prior_fit_orders["alpha_order"]
    prior_beta_order = prior_fit_orders["beta_order"]
    prior_gamma_order = prior_fit_orders["gamma_order"]
    prior_order = prior_fit_orders["order"]

    target_alpha_order = target_fit_orders["alpha_order"]
    target_beta_order = target_fit_orders["beta_order"]
    target_gamma_order = target_fit_orders["gamma_order"]
    target_order = target_fit_orders["order"]

    # First, we need to parse the solution.

    alpha_args = list(sol[0: prior_alpha_order + 1]) + [0.0] * padding_length
    alpha_args = alpha_args[: target_alpha_order + 1]

    gamma_args = (
        list(sol[prior_alpha_order + 1:
                 prior_alpha_order + prior_gamma_order + 2])
        + [0.0] * padding_length
    )
    gamma_args = gamma_args[: target_gamma_order + 1]

    cP = sol[prior_alpha_order + prior_beta_order + prior_gamma_order + 3]
    cG = sol[prior_alpha_order + prior_beta_order + prior_gamma_order + 4]

    beta_args, wiggle_args, f = parse_wiggle_args(
        sol,
        {'alpha_order': prior_alpha_order,
            'beta_order': prior_beta_order,
            'gamma_order': prior_gamma_order,
            'order': prior_order,
         }
        )

    beta_args = list(beta_args) + [0.0] * padding_length
    beta_args = beta_args[: target_beta_order + 1]

    wiggle_args = list(wiggle_args) + [0.0] * padding_length
    wiggle_args = wiggle_args[: 2 * target_order]

    return np.asarray(alpha_args
                      + gamma_args
                      + beta_args
                      + [cP, cG, f]
                      + wiggle_args)


def _legendre_basis(t: np.ndarray, n_terms: int) -> np.ndarray:
    """
    Build the Legendre polynomial basis via three-term recurrence.

    :param t: Normalized coordinate array in [-1, 1].
    :param n_terms: Number of basis functions (P_0 through P_{n_terms-1}).
    :return: Array of shape (n_terms, len(t)) where row k is P_k(t).
    """
    basis = np.empty((n_terms, len(t)), dtype=float)
    if n_terms == 0:
        return basis
    basis[0] = 1.0
    if n_terms == 1:
        return basis
    basis[1] = t
    for k in range(1, n_terms - 1):
        basis[k + 1] = ((2 * k + 1) * t * basis[k] - k * basis[k - 1]) / (k + 1)
    return basis


def build_wiggly_model(
    envelope_prototype: Callable[
        [np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]
    ],
    background_prototype: Callable[
        [np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]
    ],
    fit_orders: dict[str, int],
    prior_fit_sol: np.typing.NDArray[np.float64] | None = None,
    prior_fit_orders: dict[str, int] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
):
    """Build the full model, starting parameters, wiggle factory, bounds and Jacobian.

    For A=alpha_order, G=gamma_order, B=beta_order and H=order in fit_orders,
    parameters are [alpha_0..alpha_A, gamma_0..gamma_G, beta_0..beta_B,
    cP, cG, f, a1, b1, ..., aH, bH]; each a_j/b_j is a sine/cosine pair.
    An envelope with coefficients e_0..e_d is
    E=e_0*(1+sum(e_k*P_k(t), k=1..d)), with Legendre polynomials P_k and
    t=2*(x-x_min)/(x_max-x_min)-1. The first coefficient scales the whole
    envelope; higher coefficients set its shape, not independent amplitudes.
    x is the one-dimensional trace coordinate used in the oscillatory phase,
    not wavelength; use nonempty x and distinct limits. Omitted limits use the
    current x extrema. S=sum(a_j*sin(2*pi*j*f*x)+b_j*cos(2*pi*j*f*x), j=1..H).
    With bg=background_prototype(x), the full model is
    wm=alpha*(1+cP*beta*S)+gamma*bg*(1+cG*beta*S).
    background_prototype supplies the fixed background function bg; it need
    not have a scaled Legendre form.

    wm and wm_jac use the Legendre basis directly. The wiggle factory uses
    envelope_prototype(x, *coefficients), which must implement the same scaled
    Legendre convention and bind the same x limits for consistent evaluation.
    get_wwm(*parameters) returns the additive wiggle callable
    (alpha*cP+gamma*bg*cG)*beta*S; envelope_only=True omits S.

    Return (wm, initial_guess, get_wwm, bounds, wm_jac), with
    N=A+G+B+6+2*H parameters. initial_guess has length N; bounds has shape (2,N),
    with lower/upper rows. wm(x,*parameters) and get_wwm(*parameters)(x) return
    len(x) values; wm_jac(x,*parameters) has shape (len(x),N) in packed order.
    Both prior_fit_sol and prior_fit_orders are needed to convert a previous
    solution into initial_guess; they do not alter the model or bounds.
    """

    alpha_order = fit_orders["alpha_order"]
    beta_order = fit_orders["beta_order"]
    gamma_order = fit_orders["gamma_order"]
    order = fit_orders["order"]

    cP_idx = alpha_order + beta_order + gamma_order + 3
    cG_idx = alpha_order + beta_order + gamma_order + 4
    f_idx  = alpha_order + beta_order + gamma_order + 5
    ab_start = alpha_order + beta_order + gamma_order + 6

    _leg_cache: list = [None, None]

    def _eval_components(x, args):
        cache_key = (len(x), float(x[0]), float(x[-1]), args.tobytes())
        if _leg_cache[0] == cache_key:
            return _leg_cache[1]

        _x_min = x_min if x_min is not None else float(np.min(x))
        _x_max = x_max if x_max is not None else float(np.max(x))
        t = 2.0 * (x - _x_min) / (_x_max - _x_min) - 1.0

        ba = _legendre_basis(t, alpha_order + 1)
        bg_leg = _legendre_basis(t, gamma_order + 1)
        bb = _legendre_basis(t, beta_order + 1)

        alpha_coeffs = args[0: alpha_order + 1]
        alpha_coeffs_mod = np.concatenate([[1.0], alpha_coeffs[1:]])
        alpha = alpha_coeffs[0] * (ba * alpha_coeffs_mod[:, None]).sum(axis=0)

        gamma_coeffs = args[alpha_order + 1: alpha_order + gamma_order + 2]
        gamma_coeffs_mod = np.concatenate([[1.0], gamma_coeffs[1:]])
        gamma = gamma_coeffs[0] * (bg_leg * gamma_coeffs_mod[:, None]).sum(axis=0)

        beta_coeffs = args[alpha_order + gamma_order + 2: alpha_order + beta_order + gamma_order + 3]
        beta_coeffs_mod = np.concatenate([[1.0], beta_coeffs[1:]])
        beta_env = beta_coeffs[0] * (bb * beta_coeffs_mod[:, None]).sum(axis=0)

        cP = float(args[cP_idx])
        cG = float(args[cG_idx])
        f  = float(args[f_idx])

        sin_basis = np.empty((order, len(x)))
        cos_basis = np.empty((order, len(x)))
        dsin_basis = np.empty((order, len(x)))
        dcos_basis = np.empty((order, len(x)))
        for j in range(order):
            freq_j = 2.0 * np.pi * (j + 1) * f
            phase_j = freq_j * x
            sin_basis[j] = np.sin(phase_j)
            cos_basis[j] = np.cos(phase_j)
            dsin_basis[j] = 2.0 * np.pi * (j + 1) * x * np.cos(phase_j)
            dcos_basis[j] = -2.0 * np.pi * (j + 1) * x * np.sin(phase_j)

        ab_coeffs = args[ab_start: ab_start + 2 * order]
        a_coeffs = ab_coeffs[0::2]
        b_coeffs = ab_coeffs[1::2]

        sin_sum = (a_coeffs[:, None] * sin_basis + b_coeffs[:, None] * cos_basis).sum(axis=0)
        dsin_sum_df = (a_coeffs[:, None] * dsin_basis + b_coeffs[:, None] * dcos_basis).sum(axis=0)

        jwm_val = beta_env * sin_sum
        bg = background_prototype(x)

        result = {
            't': t, 'ba': ba, 'bg_leg': bg_leg, 'bb': bb,
            'alpha': alpha, 'gamma': gamma, 'beta_env': beta_env,
            'cP': cP, 'cG': cG, 'f': f,
            'sin_basis': sin_basis, 'cos_basis': cos_basis,
            'dsin_basis': dsin_basis, 'dcos_basis': dcos_basis,
            'sin_sum': sin_sum, 'dsin_sum_df': dsin_sum_df,
            'jwm_val': jwm_val, 'bg': bg,
        }
        _leg_cache[0] = cache_key
        _leg_cache[1] = result
        return result

    def wm(x, *args):
        c = _eval_components(x, np.asarray(args))
        return (c['alpha'] * (1.0 + c['cP'] * c['jwm_val'])
                + c['gamma'] * c['bg'] * (1.0 + c['cG'] * c['jwm_val']))

    def wm_jac(x, *args):
        c = _eval_components(x, np.asarray(args))
        alpha    = c['alpha']
        gamma    = c['gamma']
        beta_env = c['beta_env']
        cP       = c['cP']
        cG       = c['cG']
        jwm_val  = c['jwm_val']
        sin_sum  = c['sin_sum']
        bg       = c['bg']
        ba       = c['ba']
        bg_leg   = c['bg_leg']
        bb       = c['bb']
        sin_basis  = c['sin_basis']
        cos_basis  = c['cos_basis']
        dsin_sum_df = c['dsin_sum_df']

        n = len(x)
        n_params = (alpha_order + 1) + (gamma_order + 1) + (beta_order + 1) + 3 + 2 * order
        J = np.empty((n, n_params), dtype=float)

        col = 0

        alpha0 = alpha / (ba[0] if alpha_order == 0 else 1.0)
        for k in range(alpha_order + 1):
            if k == 0:
                J[:, col] = ba[0] * (1.0 + cP * jwm_val)
            else:
                J[:, col] = alpha * ba[k] * (1.0 + cP * jwm_val)
            col += 1

        gamma0 = gamma / (bg_leg[0] if gamma_order == 0 else 1.0)
        for k in range(gamma_order + 1):
            if k == 0:
                J[:, col] = bg_leg[0] * bg * (1.0 + cG * jwm_val)
            else:
                J[:, col] = gamma * bg_leg[k] * bg * (1.0 + cG * jwm_val)
            col += 1

        amp_factor = alpha * cP + gamma * bg * cG
        beta0 = beta_env / (bb[0] if beta_order == 0 else 1.0)
        for k in range(beta_order + 1):
            if k == 0:
                J[:, col] = amp_factor * bb[0] * sin_sum
            else:
                J[:, col] = amp_factor * beta_env * bb[k] * sin_sum
            col += 1

        J[:, col] = alpha * jwm_val
        col += 1

        J[:, col] = gamma * bg * jwm_val
        col += 1

        J[:, col] = amp_factor * beta_env * dsin_sum_df
        col += 1

        for j in range(order):
            J[:, col]     = amp_factor * beta_env * sin_basis[j]
            J[:, col + 1] = amp_factor * beta_env * cos_basis[j]
            col += 2

        return J

    def wwm(x, envelope_only, *args):
        jwm_inner = build_wiggle_term(
            envelope_prototype,
            *parse_wiggle_args(args, fit_orders),
        )

        alpha = envelope_prototype(x, *args[0 : alpha_order + 1])
        gamma = envelope_prototype(
            x, *args[alpha_order + 1: alpha_order + gamma_order + 2]
        )

        return alpha * (
            args[cP_idx]
            * jwm_inner(x, envelope_only=envelope_only)
        ) + gamma * background_prototype(x) * (
            args[cG_idx]
            * jwm_inner(x, envelope_only=envelope_only)
        )

    bounds = []
    initial_guess = []

    bounds.append([0.0, 2.0])
    initial_guess.append(0.5)
    for j in range(alpha_order):
        bounds.append([-1.0, 1.0])
        initial_guess.append(0.0)

    bounds.append([0.0, 2.0])
    initial_guess.append(0.5)
    for j in range(gamma_order):
        bounds.append([-1.0, 1.0])
        initial_guess.append(0.0)

    bounds.append([1.0 - 1e-10, 1.0 + 1e-10])
    initial_guess.append(1.0)
    for j in range(beta_order):
        bounds.append([-0.5, 0.5])
        initial_guess.append(0.0)

    bounds.append([1.0 - 1e-10, 1.0 + 1e-10])
    initial_guess.append(1.0)

    bounds.append([-1e-10, 1e-10])
    initial_guess.append(0.0)

    bounds.append([0.9, 1.1])
    initial_guess.append(1.0)

    for j in range(order):
        bounds.append([-1.0, 1.0])
        initial_guess.append(0.0)
        bounds.append([-1.0, 1.0])
        initial_guess.append(0.0)

    bounds = np.asarray(bounds).T

    if prior_fit_sol is not None and prior_fit_orders is not None:
        initial_guess = list(
            convert_wiggle_solution(
                prior_fit_sol,
                prior_fit_orders,
                fit_orders)
        )

    def get_wwm(*aa):
        def jwm_to_return(x, envelope_only=False):
            return wwm(x, envelope_only, *aa)

        return jwm_to_return

    return (
        wm,
        initial_guess,
        get_wwm,
        bounds,
        wm_jac,
    )


def dewiggle_lambda_space(
    trace_model,
    continuum_model,
    lam,
    target_spectrum,
    dq_spectrum,
    target_maskarray=None,
    noise_spectrum=None,
    plot=False,
    plot_label="",
    background_pf=None,
    min_fit_params={"alpha_order": 0, "beta_order": 3, "gamma_order": 0, "order": 3},
    max_fit_params={"alpha_order": 0, "beta_order": 3, "gamma_order": 0, "order": 3},
    sqrt_nu_improvement_threshold=0.0001,
    frequency_fudge=1.0,
):
    """
    Given a trace_model and a target_spectrum, dewiggle the target_spectrum by
    comparing it to the trace_model.
    """

    normalization = _eval_pf(continuum_model, lam)
    good_indices = np.isfinite(target_spectrum)

    logger.debug(
        f"The continuum model polynomial parameters are {continuum_model}",
    )
    logger.debug("The target_spectrum has ")
    logger.debug(f"    np.nanmin = {np.nanmin(target_spectrum[good_indices])}")
    logger.debug(f" and np.nanmax = {np.nanmax(target_spectrum[good_indices])}")

    normalized_spectrum = target_spectrum / normalization
    logger.debug(
        f"The normalized_spectrum has np.nanmin = "
        f"{np.nanmin(normalized_spectrum[good_indices])} and np.nanmax = "
        f"{np.nanmax(normalized_spectrum[good_indices])}"
    )

    normalization_constant = np.nanmedian(normalized_spectrum[good_indices])
    logger.debug(
        f"The input lambda run from {lam[0]} to {lam[-1]}, with min = {np.min(lam)} "
        f"and max = {np.max(lam)}"
    )
    logger.debug(f"The normalization constant is {normalization_constant}")

    normalized_spectrum /= normalization_constant
    logger.debug(
        f"The normalized_spectrum has np.nanmin = {np.nanmin(normalized_spectrum)} "
        f"and np.nanmax = {np.nanmax(normalized_spectrum)}"
    )

    good_indices = np.isfinite(normalized_spectrum)

    if dq_spectrum is None:
        dq_spectrum = np.zeros_like(target_spectrum)
    good_indices = (
        good_indices
        * (dq_spectrum[:] == 0)
        * (
            normalized_spectrum
            < np.nanpercentile(normalized_spectrum[good_indices], 99)
        )
        * (normalized_spectrum > np.nanpercentile(normalized_spectrum[good_indices], 1))
    )
    if target_maskarray is not None:
        good_indices = good_indices * target_maskarray

    med_px = trace_model[0][0](lam) * frequency_fudge
    med_px -= np.nanmedian(med_px)
    inverse_model = partial(
        np.interp,
        xp=med_px[np.argsort(med_px)],
        fp=lam[np.argsort(med_px)]
    )

    background_normalization = np.nanmedian(
        _eval_pf(background_pf, lam) / _eval_pf(continuum_model, lam)
    )

    def background_prototype_exact(x):
        if background_normalization == 0:
            return np.zeros_like(x)
        else:
            return (
            _eval_pf(background_pf, inverse_model(x))
            / _eval_pf(continuum_model, inverse_model(x))
            / background_normalization
        )

    background_prototype = background_prototype_exact

    x_vals = med_px[good_indices]

    y_for_fft = normalized_spectrum[good_indices]

    envelope = partial(generic_envelope, min_x=np.min(x_vals), max_x=np.max(x_vals))

    if noise_spectrum is not None:
        normalized_noise_spectrum = (
            noise_spectrum / normalization / normalization_constant
        )
        noise_for_fft = normalized_noise_spectrum[good_indices]
    else:
        noise_spectrum = (
            np.ones_like(target_spectrum) * normalization * normalization_constant
        )
        normalized_noise_spectrum = (
            noise_spectrum / normalization / normalization_constant
        )
        noise_for_fft = normalized_noise_spectrum[good_indices]

    fit_params = min_fit_params.copy()

    _x_min = float(np.min(x_vals))
    _x_max = float(np.max(x_vals))

    ww, ww0, jwm, bounds, wm_jac = build_wiggly_model(
        envelope,
        background_prototype,
        fit_orders=fit_params,
        x_min=_x_min,
        x_max=_x_max,
    )
    xsol = [0] * len(ww0)
    xsol[1] = 1

    reduced_x2 = 1e15
    x2 = 1e15

    def fit_wrapper(
        order=3,
        alpha_order=0,
        beta_order=3,
        gamma_order=0,
        prior_fit_sol=None,
        prior_order=3,
        prior_alpha_order=0,
        prior_beta_order=3,
        prior_gamma_order=0,
    ):
        # logger.info(
        #     f"Performing fit with (O, 𝛼, 𝛽, 𝛾) = ({order}, {alpha_order},
        # {beta_order}, {gamma_order})"
        # )
        ww, ww0, jwm, bounds, wm_jac = build_wiggly_model(
            envelope,
            background_prototype,
            fit_orders={
                "alpha_order": alpha_order,
                "beta_order": beta_order,
                "gamma_order": gamma_order,
                "order": order,
            },
            prior_fit_sol=prior_fit_sol,
            prior_fit_orders={
                "alpha_order": prior_alpha_order,
                "beta_order": prior_beta_order,
                "gamma_order": prior_gamma_order,
                "order": prior_order,
            },
            x_min=_x_min,
            x_max=_x_max,
        )

        xsol, xcov = curve_fit(
            ww,
            x_vals,
            y_for_fft,
            p0=np.asarray(ww0),
            bounds=np.asarray(bounds),
            sigma=noise_for_fft,
            jac=wm_jac,
            method='trf',
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
        )
        # logger.info('Fit model with order = %d', order)
        # logger.debug(
        #     f"Successfully fit with (O, 𝛼, 𝛽, 𝛾) = ({order}, {alpha_order},
        # {beta_order}, {gamma_order})"
        # )
        # logger.info(f'found xsol = {xsol}')
        # x2 = np.nanmean((ww(X_vals, *xsol) / y_for_fft - 1) ** 2) ** 0.5
        x2 = np.nansum(((ww(x_vals, *xsol) - y_for_fft) / noise_for_fft) ** 2)
        reduced_x2 = x2 / (len(x_vals) - len(xsol))
        # logger.info(f"  Found 𝜒2 = {x2:.3f} and √𝜈  = {reduced_x2**0.5:.3f}")

        return (xsol, xcov, x2, reduced_x2, ww, jwm)

    improving = True

    successful_fit_parameters = False
    try:
        xsol, _, x2, reduced_x2, ww, jwm = fit_wrapper(
            order=fit_params["order"],
            alpha_order=fit_params["alpha_order"],
            beta_order=fit_params["beta_order"],
            gamma_order=fit_params["gamma_order"],
        )
        successful_fit_parameters = True

    except Exception as e:

        logger.warning("Initial fit failed, reverting to default parameters")
        logger.warning(f" Exception: {e}")

        fit_params = min_fit_params.copy()

        ww, ww0, jwm, bounds, wm_jac = build_wiggly_model(
            envelope,
            background_prototype,
            fit_orders=fit_params,
            x_min=_x_min,
            x_max=_x_max,
        )
        xsol = [0] * len(ww0)
        xsol[1] = 1
        reduced_x2 = 1e15
        x2 = 1e15
        successful_fit_parameters = False


    while improving:

        improving = False

        logger.info(
            80 * "-"
        )
        logger.info(
            f"Current fit parameters are (O, 𝛼, 𝛽, 𝛾) = ({fit_params['order']}, "
            f"{fit_params['alpha_order']}, {fit_params['beta_order']}, "
            f"{fit_params['gamma_order']})"
        )
        logger.info(f"  𝜒2 = {x2:.4f} and √𝜈  = {reduced_x2**0.5:.4f}")
        logger.info(
            80 * "-"
        )
        knobs_to_turn = ["order", "beta_order", "alpha_order", "gamma_order"]
        for knob in knobs_to_turn:
            proposed_fit_params = fit_params.copy()

            if proposed_fit_params[knob] + 1 <= max_fit_params[knob]:
                proposed_fit_params[knob] += 1
                logger.info(
                    f"    Trying ({proposed_fit_params['order']}, "
                    f"{proposed_fit_params['alpha_order']}, "
                    f"{proposed_fit_params['beta_order']}, "
                    f"{proposed_fit_params['gamma_order']})"
                )
                try:
                    
                    prior_fit_sol = xsol.copy() if successful_fit_parameters else None

                    (
                        prop_xsol,
                        _,
                        prop_x2,
                        prop_reduced_x2,
                        prop_ww,
                        prop_jwm,
                    ) = fit_wrapper(
                        order=proposed_fit_params["order"],
                        alpha_order=proposed_fit_params["alpha_order"],
                        beta_order=proposed_fit_params["beta_order"],
                        gamma_order=proposed_fit_params["gamma_order"],
                        prior_fit_sol=prior_fit_sol,
                        prior_order=fit_params["order"],
                        prior_alpha_order=fit_params["alpha_order"],
                        prior_beta_order=fit_params["beta_order"],
                        prior_gamma_order=fit_params["gamma_order"],
                    )

                    logger.info(
                        f"      𝜒2 = {prop_x2:.4f} and √𝜈  = {prop_reduced_x2**0.5:.4f}"
                    )
                    if (
                        prop_reduced_x2**0.5
                        < reduced_x2**0.5 - sqrt_nu_improvement_threshold
                    ):
                        xsol = prop_xsol
                        # xcov = prop_xcov
                        x2 = prop_x2
                        reduced_x2 = prop_reduced_x2
                        ww = prop_ww
                        jwm = prop_jwm
                        fit_params = proposed_fit_params.copy()
                        logger.info(
                            f"      Adopting ({fit_params['order']}, "
                            f"{fit_params['alpha_order']}, "
                            f"{fit_params['beta_order']}, "
                            f"{fit_params['gamma_order']})"
                        )
                        improving = True
                        successful_fit_parameters = True
                        break

                except Exception as e:

                    logger.info("      !!! Fit Failed !!!")
                    logger.info(f"       exception {e}")

    ps_continuum = envelope(x_vals, *xsol[0 : fit_params["alpha_order"] + 1])

    gal_continuum = background_prototype(x_vals) * envelope(
        x_vals,
        *xsol[
            fit_params["alpha_order"]
            + 1 : fit_params["alpha_order"]
            + fit_params["gamma_order"]
            + 2
        ],
    )

    spectrum_without_continuum = y_for_fft - ps_continuum - gal_continuum
    residual_without_continuum = (
        y_for_fft - ps_continuum - gal_continuum - jwm(*xsol)(x_vals)
    )

    _fft_freq_step = 0.01
    _fft_period = 1.0 / _fft_freq_step
    _span = np.max(x_vals) - np.min(x_vals)
    _dt_nominal = _span / (100 * len(x_vals) - 1)
    _n_per_period = max(1, round(_fft_period / _dt_nominal))
    _n_fft = int(np.ceil(100 * len(x_vals) / _n_per_period)) * _n_per_period
    _dt_fft = _fft_period / _n_per_period
    fine_x_grid = np.linspace(np.min(x_vals), np.min(x_vals) + _n_fft * _dt_fft, _n_fft + 1)

    fine_grid_y_spec = np.interp(
        fine_x_grid,
        x_vals[np.argsort(x_vals)],
        spectrum_without_continuum[np.argsort(x_vals)]
    )
    fine_grid_y_res = np.interp(
        fine_x_grid,
        x_vals[np.argsort(x_vals)],
        residual_without_continuum[np.argsort(x_vals)]
    )

    X_spec = fft(fine_grid_y_spec)
    X_res = fft(fine_grid_y_res)
    n = len(X_spec)
    dt = fine_x_grid[1] - fine_x_grid[0]
    freq = np.fft.fftfreq(n, d=dt)

    freq_trim = 5

    freq_we_care_about = (freq[:] > -freq_trim) & (freq[:] < freq_trim)

    freq = freq[freq_we_care_about]
    X_spec = X_spec[freq_we_care_about]
    X_res = X_res[freq_we_care_about]

    if plot:
        plot_dewiggle_result(
            trace_model, continuum_model, lam, target_spectrum, dq_spectrum,
            saved_xsol=xsol, saved_fit_params=fit_params,
            saved_reduced_x2=reduced_x2,
            plot=plot, plot_label=plot_label,
            background_pf=background_pf, target_maskarray=target_maskarray,
            _precomputed={
                'x_vals': x_vals, 'y_for_fft': y_for_fft, 'med_px': med_px,
                'good_indices': good_indices, 'normalization': normalization,
                'normalization_constant': normalization_constant,
                'ww': ww, 'jwm': jwm, 'envelope': envelope,
                'background_prototype': background_prototype,
                'ps_continuum': ps_continuum, 'gal_continuum': gal_continuum,
                'spectrum_without_continuum': spectrum_without_continuum,
                'freq': freq, 'X_spec': X_spec, 'X_res': X_res,
            }
        )

    return (
        0,
        target_spectrum - normalization_constant * normalization * jwm(*xsol)(med_px),
        reduced_x2,
        xsol,
        fit_params,
        {'freq': freq, 'spectrum_fft': X_spec, 'residual_fft': X_res},
    )
