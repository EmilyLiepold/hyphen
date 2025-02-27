import logging

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from time import time as tt
from .helpers import get_linenumber
logger = logging.getLogger(__name__)


def build_envelope(min_x, max_x):
    def envelope(x, *leg_coefficients):
        mod_leg_coefficients = list(leg_coefficients[:])
        mod_leg_coefficients[0] = 1
        legendre = np.polynomial.Legendre(mod_leg_coefficients)
        rescaled_x = 2 * (x - min_x) / (max_x - min_x) - 1
        leg_values = leg_coefficients[0] * legendre(rescaled_x)

        return leg_values

    return envelope


def single_wiggle(x: float, order: int, a: float, b: float, f: float) -> float:
    """

    :param x: This is the x value at which the wiggle is evaluated
    :param order: This is the order of the wiggle (e.g., 1 for a first order wiggle, 2 for a second order)
    :param a: This is the amplitude of the sin term
    :param b: This is the amplitude of the cos term
    :param f: This is a fudge factor which alters the frequency of the wiggle. Typical value should be 1.0
    :return: The wiggle, evaluated at all points in x
    """

    k = 2 * order * np.pi * f
    sin_term = np.sin(k * x)
    cos_term = np.cos(k * x)

    return a * sin_term + b * cos_term


def build_wiggle_term(
    envelope_prototype: callable, envelope_args: list, wiggle_args: list, f: float = 1
) -> callable:
    """

    :param envelope_prototype: This function defines a generic envelope. It should accept a list of x values as its
    first argument, then a list of parameters which define the envelope
    :param envelope_args: This is the list of parameters which define the envelope
    :param wiggle_args: This is a list of arguments which define the wiggle. These should pairs of sin and cos
    amplitudes for each order of the wiggle.
    :param f: This is the frequency fudge factor. Nominal value should be 1.0
    :return: A callable function which will return the wiggle model evaluated at the input x values
    """

    wiggle_args = list(wiggle_args)
    order = len(wiggle_args) // 2

    assert len(wiggle_args) % 2 == 0, "The number of wiggle arguments must be even"
    assert (
        len(wiggle_args) == 2 * order
    ), "The number of wiggle arguments is not two per order..."

    def wiggle_term(x: float, envelope_only: bool = False) -> float:
        """

        :param x: A list of values where the wiggle will be evaluated
        :param envelope_only: A flag specifying whether to return the envelope or the full model
        :return: The wiggle model evaluated at the input x values (or the envelope if envelope_only is True)
        """
        sinusoidal_part = 0 * x

        # Loop through the orders of the wiggle, adding the single wiggle associated with each
        for i in range(order):
            sinusoidal_part += single_wiggle(
                x, i + 1, wiggle_args[2 * i], wiggle_args[2 * i + 1], f
            )

        # Evaluate the envelope
        envelope = envelope_prototype(x, *envelope_args)

        # Construct and return the model
        if envelope_only:
            return envelope
        else:
            return envelope * sinusoidal_part

    return wiggle_term


def parse_wiggle_args(args, alpha_order, beta_order, gamma_order, order):
    envelope_args = args[
        alpha_order + gamma_order + 2 : alpha_order + beta_order + gamma_order + 3
    ]
    wiggle_args = args[
        alpha_order
        + beta_order
        + gamma_order
        + 6 : alpha_order
        + beta_order
        + gamma_order
        + 6
        + 2 * order
    ]
    f = args[alpha_order + beta_order + gamma_order + 5]

    return envelope_args, wiggle_args, f


def convert_wiggle_solution(
    sol,
    prior_alpha_order,
    prior_beta_order,
    prior_gamma_order,
    prior_order,
    target_alpha_order,
    target_beta_order,
    target_gamma_order,
    target_order,
):

    # First, we need to parse the solution.

    alpha_args = list(sol[0 : prior_alpha_order + 1]) + [0.0] * 100
    alpha_args = alpha_args[: target_alpha_order + 1]

    gamma_args = (
        list(sol[prior_alpha_order + 1 : prior_alpha_order + prior_gamma_order + 2])
        + [0.0] * 100
    )
    gamma_args = gamma_args[: target_gamma_order + 1]

    cP = sol[prior_alpha_order + prior_beta_order + prior_gamma_order + 3]
    cG = sol[prior_alpha_order + prior_beta_order + prior_gamma_order + 4]
    beta_args, wiggle_args, f = parse_wiggle_args(
        sol, prior_alpha_order, prior_beta_order, prior_gamma_order, prior_order
    )

    beta_args = list(beta_args) + [0.0] * 100
    beta_args = beta_args[: target_beta_order + 1]

    wiggle_args = list(wiggle_args) + [0.0] * 100
    wiggle_args = wiggle_args[: 2 * target_order]

    return np.asarray(alpha_args + gamma_args + beta_args + [cP, cG, f] + wiggle_args)


def build_wiggly_model(
    envelope_prototype,
    background_prototype,
    order=3,
    alpha_order=1,
    beta_order=3,
    gamma_order=1,
    prior_fit_sol=None,
    prior_order=3,
    prior_alpha_order=1,
    prior_beta_order=3,
    prior_gamma_order=1,
):
    # if order == 0:
    #     logger.error('Order 0 is silly, why are we here? We should not be here....')

    # def jwm(x, *args, envelope_only=False):
    #     sinusoidal_part = 0 * x
    #     if order == 0:
    #         logger.error('Order 0 is silly, why are we here?')
    #     for i in range(order):
    #         sinusoidal_part += single_wiggle(
    #             x, i + 1, args[2 * i + 2], args[2 * i + 3], args[1]
    #         )
    #
    #     leg = envelope_prototype(
    #         x, 1, args[2 * order + 2], args[2 * order + 3], args[2 * order + 4]
    #     )
    #
    #     model = leg * sinusoidal_part
    #     if envelope_only:
    #         return leg * args[2]
    #     else:
    #         return model

    def wm(x, *args):
        # The list of args are:
        # args[0:alpha_order + 1] are the alpha_order + 1 coefficients which define the envelope for the PS continuum
        # args[alpha_order + 1: alpha_order + gamma_order + 2] are the gamma_order + 1 coefficients which define the envelope for the background
        # args[alpha_order + gamma_order + 2: alpha_order + beta_order + gamma_order + 3] are the beta_order + 1 coefficients which define the envelope for the wiggle
        # args[alpha_order + beta_order + gamma_order + 3] is the c_P coupling parameter between the wiggle and the PS continuum
        # args[alpha_order + beta_order + gamma_order + 4] is the c_G coupling parameter between the wiggle and the background continuum
        # args[alpha_order + beta_order + gamma_order + 5] is the f parameter which defines the frequency of the wiggle
        # args[alpha_order + beta_order + gamma_order + 6:alpha_order + beta_order + gamma_order + 6 + 2 * order - 1] are the 2 * order - 1 parameters which define the wiggle model. The amplitude for the first order is fixed to 1.

        # print()
        # logger.info(f'{alpha_order}, {beta_order}, {gamma_order}, {order}')
        # logger.info(f'{args}')
        # logger.info(f'{parse_wiggle_args(args, alpha_order, beta_order, gamma_order, order)}')
        # print()
        # print(args)

        # t0 = tt()

        # dt0 = tt() - t0
        # l0 = get_linenumber()
        
        # t1 = tt()

        jwm = build_wiggle_term(
            envelope_prototype,
            *parse_wiggle_args(args, alpha_order, beta_order, gamma_order, order),
        )

        # dt1 = tt() - t1
        # l1 = get_linenumber()
        # # print(f'T              E {dt1}; line {l1}')
        # t2 = tt()

        alpha = envelope_prototype(x, *args[0 : alpha_order + 1])
        gamma = envelope_prototype(
            x, *args[alpha_order + 1 : alpha_order + gamma_order + 2]
        )

        # dt2 = tt() - t2
        # l2 = get_linenumber()
        # # print(f'T              E {dt2}; line {l2}')
        # t3 = tt()

        jwm_term = jwm(x, envelope_only=False)

        # dt3 = tt() - t3
        # l3 = get_linenumber()
        # # print(f'T              E {dt2}; line {l2}')
        # t4 = tt()

        ta = alpha * (   
            1
            + args[alpha_order + beta_order + gamma_order + 3]
            * jwm_term
        )

        # dt4 = tt() - t4
        # l4 = get_linenumber()
        # # print(f'T              E {dt2}; line {l2}')
        # t5 = tt()

        bb = background_prototype(x)

        # dt5 = tt() - t5
        # l5 = get_linenumber()
        # # print(f'T              E {dt2}; line {l2}')
        # t6 = tt()

        tb = gamma * bb * (
            1
            + args[alpha_order + beta_order + gamma_order + 4]
            * jwm_term
        )


        # dt6 = tt() - t6
        # l6 = get_linenumber()
        # # print(f'T              E {dt2}; line {l2}')
        # t7 = tt()

        to_return = ta + tb

        # dt7 = tt() - t7
        # l7 = get_linenumber()

        # dt_total = tt() - t0
        # print(f'T              E {dt0}; {dt0 / dt_total}; line {l0}')
        # print(f'T              E {dt1}; {dt1 / dt_total}; line {l1}')
        # print(f'T              E {dt2}; {dt2 / dt_total}; line {l2}')
        # print(f'T              E {dt3}; {dt3 / dt_total}; line {l3}')
        # print(f'T              E {dt4}; {dt4 / dt_total}; line {l4}')
        # print(f'T              E {dt5}; {dt5 / dt_total}; line {l5}')
        # print(f'T              E {dt6}; {dt6 / dt_total}; line {l6}')
        # print(f'T              E {dt7}; {dt7 / dt_total}; line {l7}')
        

        
        # t4 = tt()


        return to_return

    def wwm(x, envelope_only, *args):
        # The list of args are:
        # args[0:alpha_order + 1] are the alpha_order + 1 coefficients which define the envelope for the PS continuum
        # args[alpha_order + 1: alpha_order + gamma_order + 2] are the gamma_order + 1 coefficients which define the envelope for the background
        # args[alpha_order + gamma_order + 2: alpha_order + beta_order + gamma_order + 3] are the beta_order + 1 coefficients which define the envelope for the wiggle
        # args[alpha_order + beta_order + gamma_order + 3] is the c_P coupling parameter between the wiggle and the PS continuum
        # args[alpha_order + beta_order + gamma_order + 4] is the c_G coupling parameter between the wiggle and the background continuum
        # args[alpha_order + beta_order + gamma_order + 5] is the f parameter which defines the frequency of the wiggle
        # args[alpha_order + beta_order + gamma_order + 6:alpha_order + beta_order + gamma_order + 6 + 2 * order - 1] are the 2 * order - 1 parameters which define the wiggle model. The amplitude for the first order is fixed to 1.

        jwm = build_wiggle_term(
            envelope_prototype,
            *parse_wiggle_args(args, alpha_order, beta_order, gamma_order, order),
        )

        alpha = envelope_prototype(x, *args[0 : alpha_order + 1])
        gamma = envelope_prototype(
            x, *args[alpha_order + 1 : alpha_order + gamma_order + 2]
        )

        to_return = alpha * (
            args[alpha_order + beta_order + gamma_order + 3]
            * jwm(x, envelope_only=envelope_only)
        ) + gamma * background_prototype(x) * (
            args[alpha_order + beta_order + gamma_order + 4]
            * jwm(x, envelope_only=envelope_only)
        )

        return to_return

    bounds = []
    initial_guess = []

    # alpha
    bounds.append([0, 2])
    initial_guess.append(0.5)
    for j in range(alpha_order):
        bounds.append([-1, 1])
        initial_guess.append(0.0)

    # gamma
    bounds.append([0, 2])
    initial_guess.append(0.5)
    for j in range(gamma_order):
        bounds.append([-1, 1])
        initial_guess.append(0.0)

    # beta
    bounds.append([1 - 1e-10, 1 + 1e-10])
    initial_guess.append(1.0)
    for j in range(beta_order):
        bounds.append([-0.5, 0.5])
        initial_guess.append(0.0)

    # cP
    bounds.append([1.0 - 1e-10, 1.0 + 1e-10])
    initial_guess.append(1.0)

    # cG
    bounds.append([-1e-10, 1e-10])
    initial_guess.append(0.0)

    # f
    bounds.append([0.99, 1.01])
    initial_guess.append(1.0)

    # a_1
    bounds.append([-1, 1])
    initial_guess.append(0.0)

    # b_1
    bounds.append([-1, 1])
    initial_guess.append(0.0)

    for j in range(order - 1):
        # a_j
        bounds.append([-1, 1])
        initial_guess.append(0.0)

        # b_j
        bounds.append([-1, 1])
        initial_guess.append(0.0)

    bounds = np.asarray(bounds)

    if prior_fit_sol is not None:
        initial_guess = list(
            convert_wiggle_solution(
                prior_fit_sol,
                prior_alpha_order,
                prior_beta_order,
                prior_gamma_order,
                prior_order,
                alpha_order,
                beta_order,
                gamma_order,
                order,
            )
        )

    def get_wwm(*aa):
        def jwm_to_return(x, envelope_only=False):
            return wwm(x, envelope_only, *aa)

        return jwm_to_return

    return (
        wm,
        initial_guess,
        get_wwm,
        bounds.T,
    )


def dewiggle_lambda_space(
    trace_model,
    continuum_model,
    lam,
    target_spectrum,
    dq_spectrum,
    noise_spectrum=None,
    plot=False,
    plot_label="",
    background_pf=None,
    min_fit_params={"alpha_order": 0, "beta_order": 3, "gamma_order": 0, "order": 3},
    max_fit_params={"alpha_order": 0, "beta_order": 3, "gamma_order": 0, "order": 3},
    sqrt_nu_improvement_threshold=0.001,
):
    """
    Given a trace_model and a target_spectrum, dewiggle the target_spectrum by
    comparing it to the trace_model.
    """

    t0 = tt()
    print(f'T     B {tt() - t0}; line {get_linenumber()}')
    t0 = tt()

    normalization = np.polyval(continuum_model, lam)
    good_indices = np.isfinite(target_spectrum)
    logger.debug(
        f"The continuum model polynomial parameters are {continuum_model}",
    )
    logger.debug(
        f"The target_spectrum has np.nanmin = {np.nanmin(target_spectrum[good_indices])} and np.nanmax = {np.nanmax(target_spectrum[good_indices])}"
    )

    normalized_spectrum = target_spectrum / normalization
    logger.debug(
        f"The normalized_spectrum has np.nanmin = {np.nanmin(normalized_spectrum[good_indices])} and np.nanmax = {np.nanmax(normalized_spectrum[good_indices])}"
    )

    normalization_constant = np.nanmedian(normalized_spectrum[good_indices])
    logger.debug(
        f"The input lambda run from {lam[0]} to {lam[-1]}, with min = {np.min(lam)} and max = {np.max(lam)}"
    )
    logger.debug(f"The normalization constant is {normalization_constant}")

    normalized_spectrum /= normalization_constant
    logger.debug(
        f"The normalized_spectrum has np.nanmin = {np.nanmin(normalized_spectrum)} and np.nanmax = {np.nanmax(normalized_spectrum)}"
    )

    good_indices = np.isfinite(normalized_spectrum)
    good_indices = (
        good_indices
        * (dq_spectrum[:] == 0)
        * (
            normalized_spectrum
            < np.nanpercentile(normalized_spectrum[good_indices], 99)
        )
        * (normalized_spectrum > np.nanpercentile(normalized_spectrum[good_indices], 1))
    )

    med_px = trace_model[0][0](lam)
    med_px -= np.nanmedian(med_px)
    inverse_model = interp1d(med_px, lam, fill_value="extrapolate")

    background_normalization = np.nanmedian(
        np.polyval(background_pf, lam) / np.polyval(continuum_model, lam)
    )

    def background_prototype_exact(x):
        return (
            np.polyval(background_pf, inverse_model(x))
            / np.polyval(continuum_model, inverse_model(x))
            / background_normalization
        )

    # print(background_pf)
    # exit()
    background_prototype_pf = np.polyfit(med_px, background_prototype_exact(med_px), len(background_pf) - 1)

    def background_prototype(x):
        return np.polyval(background_prototype_pf, x)

    fig, axes = plt.subplots(1)

    axes.plot(med_px, background_prototype_exact(med_px), label="Background Prototype")
    axes.plot(med_px, background_prototype(med_px), label="Background Prototype PF")

    plt.show()

    print(f'T     B {tt() - t0}; line {get_linenumber()}')
    t0 = tt()

    x_vals = med_px[good_indices]

    y_for_fft = normalized_spectrum[good_indices]

    envelope = build_envelope(np.min(x_vals), np.max(x_vals))

    print(f'T     B {tt() - t0}; line {get_linenumber()}')
    t0 = tt()

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

    print(f'T     B {tt() - t0}; line {get_linenumber()}')
    t0 = tt()

    ## Build a baseline model with the minimum fit parameters
    ww, ww0, jwm, bounds = build_wiggly_model(
        envelope,
        background_prototype,
        order=fit_params["order"],
        alpha_order=fit_params["alpha_order"],
        gamma_order=fit_params["gamma_order"],
        beta_order=fit_params["beta_order"],
    )
    xsol = [0] * len(ww0)
    xsol[1] = 1

    print(f'T     B {tt() - t0}; line {get_linenumber()}')
    prev_line = get_linenumber()
    t0 = tt()

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
        # t0 = tt()
        # print(f'T       C {tt() - t0}; line {get_linenumber()}')
        # t0 = tt()

        logger.info(
            f"Performing fit with (O, 𝛼, 𝛽, 𝛾) = ({order}, {alpha_order}, {beta_order}, {gamma_order})"
        )

        # print(f'T       C {tt() - t0}; line {get_linenumber()}')
        # t0 = tt()
        ww, ww0, jwm, bounds = build_wiggly_model(
            envelope,
            background_prototype,
            order=order,
            alpha_order=alpha_order,
            gamma_order=gamma_order,
            beta_order=beta_order,
            prior_fit_sol=prior_fit_sol,
            prior_order=prior_order,
            prior_alpha_order=prior_alpha_order,
            prior_beta_order=prior_beta_order,
            prior_gamma_order=prior_gamma_order,
        )

        # print(f'T       C {tt() - t0}; line {get_linenumber()}')
        # t0 = tt()

        # logger.info(f'Initiated wiggle model with order = {order}, alpha = {alpha_order}, beta = {beta_order}, gamma = {gamma_order}')
        # print(
        #     f"Failed to fit with order = {current_order}, alpha_order = {alpha_order}, beta_order = {beta_order}, gamma_order = {gamma_order}")
        xsol, xcov = curve_fit(
            ww,
            x_vals,
            y_for_fft,
            p0=np.asarray(ww0),
            bounds=np.asarray(bounds),
            sigma=noise_for_fft,
        )

        # print(f'T       C {tt() - t0}; line {get_linenumber()}')
        # t0 = tt()
        # logger.info('Fit model with order = %d', order)
        logger.debug(
            f"Successfully fit with (O, 𝛼, 𝛽, 𝛾) = ({order}, {alpha_order}, {beta_order}, {gamma_order})"
        )
        # print(f'T       C {tt() - t0}; line {get_linenumber()}')
        # t0 = tt()
        # logger.info(f'found xsol = {xsol}')
        # x2 = np.nanmean((ww(X_vals, *xsol) / y_for_fft - 1) ** 2) ** 0.5
        x2 = np.nansum(((ww(x_vals, *xsol) - y_for_fft) / noise_for_fft) ** 2)
        reduced_x2 = x2 / (len(x_vals) - len(xsol))
        logger.info(f"  Found 𝜒2 = {x2:.3f} and √𝜈  = {reduced_x2**0.5:.3f}")

        # print(f'T       C {tt() - t0}; line {get_linenumber()}')
        # t0 = tt()

        return (xsol, xcov, x2, reduced_x2, ww, jwm)

    print(f'T     B {tt() - t0}; lines {prev_line}-{get_linenumber()}')
    prev_line = get_linenumber()
    t0 = tt()

    improving = True

    try:
        print(f'T     B {tt() - t0}; line {get_linenumber()}')
        t0 = tt()

        xsol, xcov, x2, reduced_x2, ww, jwm = fit_wrapper(
            order=fit_params["order"],
            alpha_order=fit_params["alpha_order"],
            beta_order=fit_params["beta_order"],
            gamma_order=fit_params["gamma_order"],
        )
        print(f'T     B {tt() - t0}; line {get_linenumber()}')
        t0 = tt()

    except Exception:

        fit_params["alpha_order"] = 0
        fit_params["beta_order"] = 3
        fit_params["gamma_order"] = 0
        fit_params["order"] = 3

        ww, ww0, jwm, bounds = build_wiggly_model(
            envelope,
            background_prototype,
            order=3,
            alpha_order=0,
            beta_order=3,
            gamma_order=0,
        )
        xsol = [0] * len(ww0)
        xsol[1] = 1
        reduced_x2 = 1e15
        x2 = 1e15

        print(f'T     B {tt() - t0}; line {get_linenumber()}')
        t0 = tt()

    while improving:

        improving = False

        knobs_to_turn = ["order", "beta_order", "alpha_order", "gamma_order"]
        for knob in knobs_to_turn:

            print(f'T     B {tt() - t0}; line {get_linenumber()}')
            t0 = tt()
            proposed_fit_params = fit_params.copy()

            if proposed_fit_params[knob] + 1 <= max_fit_params[knob]:
                proposed_fit_params[knob] += 1
                try:
                    # print(f'T     D {tt() - t0}; line {get_linenumber()}')
                    # t0 = tt()
                    (
                        prop_xsol,
                        prop_xcov,
                        prop_x2,
                        prop_reduced_x2,
                        prop_ww,
                        prop_jwm,
                    ) = fit_wrapper(
                        order=proposed_fit_params["order"],
                        alpha_order=proposed_fit_params["alpha_order"],
                        beta_order=proposed_fit_params["beta_order"],
                        gamma_order=proposed_fit_params["gamma_order"],
                        prior_fit_sol=xsol,
                        prior_order=fit_params["order"],
                        prior_alpha_order=fit_params["alpha_order"],
                        prior_beta_order=fit_params["beta_order"],
                        prior_gamma_order=fit_params["gamma_order"],
                    )
                    # print(f'T     D {tt() - t0}; line {get_linenumber()}')
                    # t0 = tt()

                    if (
                        prop_reduced_x2**0.5
                        < reduced_x2**0.5 - sqrt_nu_improvement_threshold
                    ):
                        xsol = prop_xsol
                        xcov = prop_xcov
                        x2 = prop_x2
                        reduced_x2 = prop_reduced_x2
                        ww = prop_ww
                        jwm = prop_jwm
                        fit_params["order"] += 1
                        fit_params = proposed_fit_params
                        logger.info(
                            f"Found improvement with (O, 𝛼, 𝛽, 𝛾) = ({fit_params['order']}, {fit_params['alpha_order']}, {fit_params['beta_order']}, {fit_params['gamma_order']})"
                        )
                        improving = True
                        continue
                    print(f'T     B {tt() - t0}; line {get_linenumber()}')
                    t0 = tt()
                except:
                    logger.debug(
                        f"Failed to fit with (O, 𝛼, 𝛽, 𝛾) = ({proposed_fit_params['order']}, {proposed_fit_params['alpha_order']}, {proposed_fit_params['beta_order']}, {proposed_fit_params['gamma_order']})"
                    )

    print(f'T     B {tt() - t0}; line {get_linenumber()}')
    t0 = tt()
    logger.info("Fitting complete. Now to plotting. ")
    # logger.info(f'{fit_params}')
    # logger.info(f'{xsol}')
    if plot:

        envelope_args, wiggle_args, f = parse_wiggle_args(
            xsol,
            fit_params["alpha_order"],
            fit_params["beta_order"],
            fit_params["gamma_order"],
            fit_params["order"],
        )

        print(f'T     B {tt() - t0}; line {get_linenumber()}')
        t0 = tt()
        fig, axes = plt.subplots(5, 1, figsize=(10, 16))

        fig.suptitle(
            f"{plot_label}\nx2 = {x2:.3f}, sqrt(nu) = {reduced_x2**0.5:.3f}, f = {f:.5f}"
        )
        # fig.suptitle(f'x2 = {x2:.2f}')
        axes[0].set_ylabel("Flux, arbitrary")
        axes[0].plot(lam, target_spectrum, label="Input Spectrum", c="k")
        # axes[0].fill_between(lam, target_spectrum - noise_spectrum, target_spectrum + noise_spectrum, alpha=0.2, color="k")
        axes[0].plot(
            lam,
            target_spectrum
            - normalization_constant * normalization * jwm(*xsol)(med_px),
            label="Dewiggled Spectrum",
            c="r",
        )
        axes[0].legend(fontsize=6)


        axes[1].plot(x_vals, y_for_fft, label="Input Spectrum", c="k")
        axes[1].plot(x_vals, ww(x_vals, *xsol), label="Full Model", c="g")

        ps_continuum = envelope(x_vals, *xsol[0 : fit_params["alpha_order"] + 1])

        axes[1].plot(
            x_vals,
            ps_continuum,
            label=f"PS Continuum model, alpha={xsol[0]:.2f}",
            c="0.5",
            linestyle=":",
        )

        gal_continuum = background_prototype(x_vals) * envelope(
            x_vals,
            *xsol[
                fit_params["alpha_order"]
                + 1 : fit_params["alpha_order"]
                + fit_params["gamma_order"]
                + 2
            ],
        )

        axes[1].plot(
            x_vals,
            gal_continuum,
            # xsol[-1]
            # * np.polyval(background_pf, inverse_model(X_vals))
            # / np.polyval(continuum_model, inverse_model(X_vals)),
            label=f"Gal Continuum model, gamma={xsol[1]:.2f}",
            c="0.5",
            linestyle="dashed",
        )
        axes[1].plot(x_vals, jwm(*xsol)(x_vals), label="Wiggle Model", c="b")
        envelope_args_str = ", ".join([f"{x:.2f}" for x in envelope_args])
        axes[1].plot(
            x_vals,
            jwm(*xsol)(x_vals, envelope_only=True),
            label=f"Wiggle Envelope,  l=({envelope_args_str})",
            c="b",
            linestyle=":",
        )
        axes[1].plot(
            x_vals, -jwm(*xsol)(x_vals, envelope_only=True), c="b", linestyle=":"
        )

        axes[1].plot(
            x_vals, y_for_fft - jwm(*xsol)(x_vals), label="Dewiggled Spectrum", c="r"
        )
        axes[1].legend(fontsize=6)
        axes[1].set_ylabel("Flux, normalized by central continuum")

        axes[2].set_ylabel("Residual")

        axes[2].plot(
            x_vals,
            y_for_fft - ww(x_vals, *xsol),
            label="Residual (black - green)",
            c="r",
        )

        spectrum_without_continuum = y_for_fft - ps_continuum - gal_continuum
        residual_without_continuum = (
            y_for_fft - ps_continuum - gal_continuum - jwm(*xsol)(x_vals)
        )

        axes[2].plot(
            x_vals,
            spectrum_without_continuum,
            label="Spectrum without continuum",
            c="k",
        )

        ax_ylim = np.nanmax(np.abs(axes[2].get_ylim()))
        axes[2].set_ylim(-ax_ylim, ax_ylim)

        axes[2].legend(fontsize=6)

        axes[2].axhline(-(reduced_x2**0.5), c="0.5", linestyle="--")
        axes[2].axhline(0, c="0.5", linestyle="--")
        axes[2].axhline(reduced_x2**0.5, c="0.5", linestyle="--")

        # axes[3].plot(X_vals, residual_without_continuum, label="Wiggle without continuum", c="b")

        spectrum_without_continuum_interpolation = interp1d(
            x_vals, spectrum_without_continuum, fill_value="extrapolate"
        )

        residual_without_continuum_interpolation = interp1d(
            x_vals, residual_without_continuum, fill_value="extrapolate"
        )

        fine_x_grid = np.linspace(np.min(x_vals), np.max(x_vals), 20 * len(x_vals))

        fine_grid_y_spec = spectrum_without_continuum_interpolation(fine_x_grid)
        fine_grid_y_res = residual_without_continuum_interpolation(fine_x_grid)

        X_spec = fft(fine_grid_y_spec)
        X_res = fft(fine_grid_y_res)
        n = len(X_spec)

        dt = fine_x_grid[1] - fine_x_grid[0]

        freq = np.fft.fftfreq(n, d=dt)

        freq_trim = 5
        freq_we_care_about = (freq[:] > -freq_trim  ) & (freq[:] < freq_trim)

        axes[3].stem(freq[freq_we_care_about], (np.abs(X_spec) / np.nanmax(np.abs(X_spec)))[freq_we_care_about], "k")
        axes[3].stem(freq[freq_we_care_about], (np.abs(X_res) / np.nanmax(np.abs(X_spec)))[freq_we_care_about], "r")

        axes[3].set_xlim(-freq_trim, freq_trim)
        axes[3].set_ylim(0, 1.0)
        # axes[3].set_yscale('log')

        axes[4].set_ylabel("Flux, arbitrary")
        axes[4].plot(lam, target_spectrum, label="Input Spectrum", c="k")
        # axes[2].plot(lam[good_indices], y_for_fft * normalization[good_indices], label='Input Spectrum',c='k')
        # axes[2].plot(X_vals, y_for_fft, label='Input Spectrum')
        axes[4].plot(
            lam,
            normalization_constant * normalization * ww(med_px, *xsol),
            label="Full Model",
            c="g",
        )
        axes[4].plot(
            lam,
            normalization_constant * normalization * jwm(*xsol)(med_px),
            label="Wiggle Model",
            c="b",
        )
        axes[4].plot(
            lam,
            target_spectrum
            - normalization_constant * normalization * jwm(*xsol)(med_px),
            label="Dewiggled Spectrum",
            c="r",
        )
        axes[4].legend(fontsize=6)

        # print(xsol, alpha_order, beta_order, gamma_order, current_order)

        try:
            # print(jwm(*xsol)(med_px)[good_indices])
            # print(normalization_constant)
            # print(normalization)
            # print(np.nanmin((normalization_constant * normalization * jwm(*xsol)(med_px))[good_indices]),
            #     np.nanmax(target_spectrum[good_indices]))
            axes[4].set_ylim(
                np.nanmin(
                    (normalization_constant * normalization * jwm(*xsol)(med_px))[
                        good_indices
                    ]
                ),
                np.nanmax(target_spectrum[good_indices]),
            )
        except:
            axes[4].set_ylim(0, np.nanmax(target_spectrum[good_indices]))

        axes[4].set_xlim(axes[0].get_xlim())
        xlim = axes[1].get_xlim()
        low_x = int(np.floor(xlim[0]))
        high_x = int(np.ceil(xlim[1]))

        axes[1].set_xticks(np.arange(low_x, high_x, 1))
        axes[1].grid()
        print(f'T     B {tt() - t0}; line {get_linenumber()}')
        t0 = tt()
        plot.savefig()
        print(f'T     B {tt() - t0}; line {get_linenumber()}')
        t0 = tt()
        plt.close()

    return (
        0,
        target_spectrum - normalization_constant * normalization * jwm(*xsol)(med_px),
        reduced_x2,
        xsol,
        fit_params,
    )
