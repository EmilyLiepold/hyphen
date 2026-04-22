import logging

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.optimize import curve_fit
from typing import Callable
from functools import partial
logger = logging.getLogger(__name__)


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
):

    alpha_order = fit_orders["alpha_order"]
    beta_order = fit_orders["beta_order"]
    gamma_order = fit_orders["gamma_order"]
    order = fit_orders["order"]

    def wm(x, *args):
        # The list of args are:
        # args[0:alpha_order + 1] are the alpha_order + 1 coefficients which
        #   define the envelope for the PS continuum
        # args[alpha_order + 1: alpha_order + gamma_order + 2] are the
        #   gamma_order + 1 coefficients which define the envelope for the
        #   background
        # args[alpha_order + gamma_order + 2: alpha_order + beta_order +
        #   gamma_order + 3] are the beta_order + 1 coefficients which define
        #   the envelope for the wiggle
        # args[alpha_order + beta_order + gamma_order + 3] is the c_P coupling
        #   parameter between the wiggle and the PS continuum
        # args[alpha_order + beta_order + gamma_order + 4] is the c_G coupling
        #   parameter between the wiggle and the background continuum
        # args[alpha_order + beta_order + gamma_order + 5] is the f parameter
        #   which defines the frequency of the wiggle
        # args[alpha_order + beta_order + gamma_order + 6:alpha_order +
        #   beta_order + gamma_order + 6 + 2 * order - 1] are the
        #   2 * order - 1 parameters which define the wiggle model.
        #   The amplitude for the first order is fixed to 1.

        jwm = build_wiggle_term(
            envelope_prototype,
            *parse_wiggle_args(args, fit_orders),
        )

        alpha = envelope_prototype(x, *args[0: alpha_order + 1])
        gamma = envelope_prototype(
            x, *args[alpha_order + 1: alpha_order + gamma_order + 2]
        )

        return alpha * (
            1
            + args[alpha_order + beta_order + gamma_order + 3]
            * jwm(x, envelope_only=False)
        ) + gamma * background_prototype(x) * (
            1
            + args[alpha_order + beta_order + gamma_order + 4]
            * jwm(x, envelope_only=False)
        )

    def wwm(x, envelope_only, *args):
        # The list of args are:
        # args[0:alpha_order + 1] are the alpha_order + 1 coefficients which define the
        # envelope for the PS continuum
        # args[alpha_order + 1: alpha_order + gamma_order + 2] are the gamma_order + 1
        # coefficients which define the envelope for the background
        # args[alpha_order + gamma_order + 2: alpha_order + beta_order + gamma_order +
        # 3] are the beta_order + 1 coefficients which define the envelope for the
        # wiggle
        # args[alpha_order + beta_order + gamma_order + 3] is the c_P coupling
        # parameter between the wiggle and the PS continuum
        # args[alpha_order + beta_order + gamma_order + 4] is the c_G coupling
        # parameter between the wiggle and the background continuum
        # args[alpha_order + beta_order + gamma_order + 5] is the f parameter which
        # defines the frequency of the wiggle
        # args[alpha_order + beta_order + gamma_order + 6:alpha_order + beta_order +
        # gamma_order + 6 + 2 * order - 1] are the 2 * order - 1 parameters which
        # define the wiggle model. The amplitude for the first order is fixed to 1.

        jwm = build_wiggle_term(
            envelope_prototype,
            *parse_wiggle_args(args, fit_orders),
        )

        alpha = envelope_prototype(x, *args[0 : alpha_order + 1])
        gamma = envelope_prototype(
            x, *args[alpha_order + 1: alpha_order + gamma_order + 2]
        )

        return alpha * (
            args[alpha_order + beta_order + gamma_order + 3]
            * jwm(x, envelope_only=envelope_only)
        ) + gamma * background_prototype(x) * (
            args[alpha_order + beta_order + gamma_order + 4]
            * jwm(x, envelope_only=envelope_only)
        )

    bounds = []
    initial_guess = []

    # alpha
    bounds.append([0.0, 2.0])
    initial_guess.append(0.5)
    for j in range(alpha_order):
        bounds.append([-1.0, 1.0])
        initial_guess.append(0.0)

    # gamma
    bounds.append([0.0, 2.0])
    initial_guess.append(0.5)
    for j in range(gamma_order):
        bounds.append([-1.0, 1.0])
        initial_guess.append(0.0)

    # beta
    bounds.append([1.0 - 1e-10, 1.0 + 1e-10])
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

    for j in range(order):
        # a_j
        bounds.append([-1.0, 1.0])
        initial_guess.append(0.0)

        # b_j
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
):
    """
    Given a trace_model and a target_spectrum, dewiggle the target_spectrum by
    comparing it to the trace_model.
    """

    normalization = np.polyval(continuum_model, lam)
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

    med_px = trace_model[0][0](lam)
    med_px -= np.nanmedian(med_px)
    inverse_model = partial(
        np.interp,
        xp=med_px[np.argsort(med_px)],
        fp=lam[np.argsort(med_px)]
    )

    background_normalization = np.nanmedian(
        np.polyval(background_pf, lam) / np.polyval(continuum_model, lam)
    )

    def background_prototype_exact(x):
        if background_normalization == 0:
            return np.zeros_like(x)
        else:
            return (
            np.polyval(background_pf, inverse_model(x))
            / np.polyval(continuum_model, inverse_model(x))
            / background_normalization
        )

    background_prototype_pf = np.polyfit(
        med_px[good_indices],
        background_prototype_exact(med_px[good_indices]),
        len(background_pf) + 4,
    )

    def background_prototype(x):
        return np.polyval(background_prototype_pf, x)

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

    # Build a baseline model with the minimum fit parameters
    ww, ww0, jwm, bounds = build_wiggly_model(
        envelope,
        background_prototype,
        fit_orders=fit_params,
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
        ww, ww0, jwm, bounds = build_wiggly_model(
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
        )

        # logger.info(f'Initiated wiggle model with
        # order = {order}, alpha = {alpha_order}, beta = {beta_order},
        # gamma = {gamma_order}')
        # print(
        #     f"Failed to fit with order = {current_order}, alpha_order = {alpha_order},
        #  beta_order = {beta_order}, gamma_order = {gamma_order}")
        xsol, xcov = curve_fit(
            ww,
            x_vals,
            y_for_fft,
            p0=np.asarray(ww0),
            bounds=np.asarray(bounds),
            sigma=noise_for_fft,
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

    try:
        xsol, _, x2, reduced_x2, ww, jwm = fit_wrapper(
            order=fit_params["order"],
            alpha_order=fit_params["alpha_order"],
            beta_order=fit_params["beta_order"],
            gamma_order=fit_params["gamma_order"],
        )
    except Exception as e:

        logger.warning("Initial fit failed, reverting to default parameters")
        logger.warning(f" Exception: {e}")

        fit_params["alpha_order"] = 0
        fit_params["beta_order"] = 1
        fit_params["gamma_order"] = 0
        fit_params["order"] = 1

        ww, ww0, jwm, bounds = build_wiggly_model(
            envelope,
            background_prototype,
            fit_orders=fit_params,
        )
        xsol = [0] * len(ww0)
        xsol[1] = 1
        reduced_x2 = 1e15
        x2 = 1e15

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
                        prior_fit_sol=xsol,
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

    fine_x_grid = np.linspace(np.min(x_vals), np.max(x_vals), 20 * len(x_vals))

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
        logger.debug("Beginning plots")
        envelope_args, wiggle_args, f = parse_wiggle_args(
            xsol,
            fit_params,
        )

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

        axes2[0].plot(x_vals, y_for_fft, label="Input Spectrum", c="k",
                      linewidth=0.5,
                      )
        axes2[0].plot(x_vals, ww(x_vals, *xsol), label="Full Model", c="g",
                      linewidth=0.5,
                      )
        axes2[0].plot(
            x_vals,
            ps_continuum,
            label=f"Central component: alpha={xsol[0]:.2f}",
            c="0.5",
            linestyle=":",
            linewidth=0.5,
        )
        axes2[0].plot(
            x_vals,
            gal_continuum,
            # xsol[-1]
            # * np.polyval(background_pf, inverse_model(X_vals))
            # / np.polyval(continuum_model, inverse_model(X_vals)),
            label=f"Background component: gamma={xsol[1]:.2f}",
            c="0.5",
            linestyle="dashed",
            linewidth=0.5,
        )
        envelope_args_str = ", ".join([f"{x:.2f}" for x in envelope_args])
        axes2[0].plot(x_vals, jwm(*xsol)(x_vals),
                      linewidth=0.5,
                      label=f"Wiggle Model\nl=({envelope_args_str})", c="b")
        
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
        axes2[0].plot(
            x_vals, y_for_fft - jwm(*xsol)(x_vals), label="Dewiggled Spectrum", c="r",
            linewidth=0.5,
        )

        handles, labels = axes2[0].get_legend_handles_labels()
        order = [0,1,4,5,2,3]  # desired order of legend entries (indices into handles/labels)
        axes2[0].legend([handles[i] for i in order], [labels[i] for i in order],
                        fontsize=6, loc='lower center', framealpha=1.0)

        axes2[0].set_ylabel("Flux, normalized (arbitrary)")
        axes2[0].set_xlabel('Detector pixel (arbitrary offset)')

        axes2[1].plot(
            x_vals,
            y_for_fft - ww(x_vals, *xsol),
            label="Dewiggled spectrum minus continuum",
            c="r",
            linewidth=0.5,
        )
        axes2[1].plot(
            x_vals,
            spectrum_without_continuum,
            label="Input spectrum minus continuum",
            c="k",
            linewidth=0.5,
        )

        ax_ylim = np.nanmax(np.abs(axes2[1].get_ylim()))
        axes2[1].set_ylim(-ax_ylim, ax_ylim)

        axes2[1].set_ylabel("Residual")
        axes2[1].set_xlabel('Detector pixel (arbitrary offset)')
        axes2[1].axhline(-(reduced_x2**0.5), c="0.5", linestyle="--",label='5% residual', zorder=-10)
        axes2[1].axhline(0, c="0.5", linestyle="--", zorder=-10)
        axes2[1].axhline(reduced_x2**0.5, c="0.5", linestyle="--", zorder=-10)
        axes2[1].legend(fontsize=6, loc='lower center')

        axes2[2].stem(
            freq,
            (np.abs(X_spec) / np.nanmax(np.abs(X_spec))),
            "k",
        )
        axes2[2].stem(
            freq,
            (np.abs(X_res) / np.nanmax(np.abs(X_spec))),
            "r",
        )
        axes2[2].scatter([], [], c='k', label='Input spectrum minus continuum')
        axes2[2].scatter([], [], c='r', label='Dewiggled spectrum minus continuum')

        axes2[2].set_xlim(0, freq_trim)
        axes2[2].set_ylim(0, 1.0)

        axes2[2].set_ylabel('Fourier spectrum')
        axes2[2].set_xlabel('Frequency (1 / detector pixels)')

        axes2[2].legend(fontsize=6, loc='upper right')
        axes.set_ylabel("Flux, arbitrary")
        axes.plot(lam, target_spectrum, label="Input Spectrum", c="k", linewidth=0.5,)
        axes.plot(
            lam,
            normalization_constant * normalization * ww(med_px, *xsol),
            label="Full Model",
            c="g",
            linewidth=0.5,
        )
        axes.plot(
            lam,
            normalization_constant * normalization * jwm(*xsol)(med_px),
            label="Wiggle Model",
            c="b",
            linewidth=0.5,
        )
        axes.plot(
            lam,
            target_spectrum
            - normalization_constant * normalization * jwm(*xsol)(med_px),
            label="Dewiggled Spectrum",
            c="r",
            linewidth=0.5,
        )
        axes.legend(fontsize=6, loc='lower center', framealpha=1.0)
        axes.set_xlabel("Wavelength (micron)")
        try:
            axes.set_ylim(
                np.nanmin(
                    (normalization_constant * normalization * jwm(*xsol)(med_px))[
                        good_indices
                    ]
                ),
                np.nanmax(target_spectrum[good_indices]),
            )
        except Exception:
            axes.set_ylim(0, np.nanmax(target_spectrum[good_indices]))

        # axes.set_xlim(axes[0].get_xlim())

        lambda_space_xlim = (np.min(lam[good_indices]), np.max(lam[good_indices]))
        x_space_xlim = (np.max(x_vals), np.min(x_vals))

        print(f"Lambda space xlim: {lambda_space_xlim}")

        axes.set_xlim(lambda_space_xlim)

        axes2[0].set_xlim(x_space_xlim)
        axes2[1].set_xlim(x_space_xlim)
        print(f"X space xlim: {x_space_xlim}")
        xlim = axes2[0].get_xlim()
        low_x = int(np.floor(xlim[0]))
        high_x = int(np.ceil(xlim[1]))
        axes2[0].set_xticks(np.arange(low_x, high_x, 1))
        axes2[0].grid()
        axes2[1].set_xticks(np.arange(low_x, high_x, 1))
        axes2[1].grid(axis='x')

        plot.savefig(fig)
        plot.savefig(fig2)
        plt.close(fig)
        plt.close(fig2)

    return (
        0,
        target_spectrum - normalization_constant * normalization * jwm(*xsol)(med_px),
        reduced_x2,
        xsol,
        fit_params,
        {'freq': freq, 'spectrum_fft': X_spec, 'residual_fft': X_res},
    )
