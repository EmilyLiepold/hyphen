import logging
import argparse

from astropy.io.fits import HDUList
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from stdatamodels.jwst.datamodels import ImageModel
from jwst.assign_wcs import nrs_ifu_wcs
from joblib import delayed
import dill as d
from typing import Callable, Optional

from .tracer_helpers import find_bars, infer_fractional_position
from .dewiggle_lambda_space import dewiggle_lambda_space, parse_wiggle_args
from .yaml_helpers import ingest_yaml
from .helpers import ProgressParallel

logger = logging.getLogger(__name__)


def load_frames(fn_2d: str, fn_3d: str) -> tuple[HDUList, HDUList, str, str]:
    """
    Load the 2D and 3D frames

    :param fn_2d: This is the filename for the 2D detector-frame fits file
    :param fn_3d: This is the filename for the 3D rectified fits file
    :return: A tuple containing the 2D and 3D frames and their shortnames
    """

    # Load the 2D and 3D frames
    frame_2d = fits.open(fn_2d)
    frame_3d = fits.open(fn_3d)

    # Define a shortname from the filename without the full path or suffix.
    fn_3d_shortname = fn_3d.split("/")[-1]
    fn_2d_shortname = fn_2d.split("/")[-1]

    logger.info(f"Using 2D frame located at {fn_2d}")
    logger.debug(f"The 2D frame has shortname {fn_2d_shortname}")
    logger.info(f"Using 3D frame located at {fn_3d}")
    logger.debug(f"The 3D frame has shortname {fn_3d_shortname}")

    return frame_2d, frame_3d, fn_2d_shortname, fn_3d_shortname


def get_wcs(
    fn_2d: str, frame_3d: HDUList, use_cached_wcs: bool = True
) -> tuple[list[WCS], WCS]:
    """
    Get the wcs for the 2D and 3D frames

    :param fn_2d: This is the filename for the 2D detector-frame fits file
    :param frame_3d: This is the fits object for the 3D detector-frame fits file.
    :param use_cached_wcs: If True, we'll try to use a cached WCS file. If False, we'll
        compute the WCS from scratch. If there's an issue with loading the cached WCS,
        we'll compute the WCS from scratch.
    :return: A tuple containing a list of the WCS for each slice in the 2D frame and
        the WCS for the 3D frame
    """
    logger.info("Computing WCS for the slices in the 2D frame")

    wcs_cachefilename = fn_2d.replace(".fits", ".wcs")

    # Compute the wcs for the 2D frame (list of wcs for each slice) and for the 3D cube
    recompute_wcs = False
    if use_cached_wcs:
        logger.info(f"Trying to use cached WCS located at {wcs_cachefilename}")
        try:
            frame_2d_wcs = d.load(open(wcs_cachefilename, "rb"))
        except FileNotFoundError:
            logger.warning("Cached WCS not found. Computing from scratch")
            recompute_wcs = True
    else:
        recompute_wcs = True

    if recompute_wcs:
        frame_2d_wcs = nrs_ifu_wcs(ImageModel(fn_2d))
        d.dump(frame_2d_wcs, open(wcs_cachefilename, "wb"))

    frame_2d_wcs = d.load(open(wcs_cachefilename, "rb"))

    # sort these slices by the lower value in the pixel_bounds
    frame_2d_wcs = sorted(frame_2d_wcs, key=lambda x: x.pixel_bounds[1][0])

    logger.info("Computing WCS for the 3D frame")

    frame_3d_wcs = WCS(frame_3d[1].header)

    return frame_2d_wcs, frame_3d_wcs


def determine_detector_name(frame_2d: HDUList) -> str:
    """
    Determine the detector from the 2D frame
    :param frame_2d: This is fits object for the 2D detector-frame exposure
    :return: The name of the detector used in the 2D frame (e.g., NRS1, NRS2)
    """
    detector = frame_2d[0].header["DETECTOR"]

    logger.debug(f"The 2D frame appears to use detector {detector}")

    return detector


def get_trace_model(
    detector_wcs: WCS, ra: float, dec: float
) -> tuple[
    Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]],
    Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]
]:
    """
    Get the trace model for a given ra and dec

    :param detector_wcs: The WCS for the detector-frame exposure
    :param ra: The RA in degrees to use for the spectral trace
    :param dec: The DEC in degrees to use for the spectral trace
    :return:
    """

    def generic_trace(
        x: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        return np.atleast_2d(detector_wcs.transform("world", "detector", ra, dec, x))

    def trace_x(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return generic_trace(x)[1, :]

    def trace_y(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return generic_trace(x)[0, :]

    return trace_x, trace_y


def get_principal_axes(
    frame_3d: HDUList
    ) -> tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64]
        ]:
    """
    Get the principal axes of the 3D frame
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    :param frame_3d: This is the fits object for the 3D detector-frame exposure
    :return: A tuple containing three numpy arrays:
        - axis_pixel: The pixel axis values.
        - axis_slice: The slice axis values.
        - axis_spectral: The spectral axis values.
    """

    # Define the axes of the 3D frame
    # Note that the ordering of the axes is opposite of the fits datacube.
    # e.g., frame_3d[1].data.shape = (NAXIS3, NAXIS2, NAXIS1)
    # axis_pixel runs along the pixel axis (e.g., each index here corresponds to a
    # different pixel)
    # axis_slice runs along the slice axis (e.g., each index here corresponds to a
    # different slice)
    # axis_spectral runs along the spectral axis

    # Note that the sign of displacements along the slice axis are reversed here for
    # convenience while plotting.
    axis_pixel = frame_3d[1].header["CDELT1"] * np.arange(frame_3d[1].header["NAXIS1"])
    axis_slice = -frame_3d[1].header["CDELT2"] * np.arange(frame_3d[1].header["NAXIS2"])
    axis_spectral = frame_3d[1].header["CRVAL3"] + frame_3d[1].header["CDELT3"] * (
        np.arange(frame_3d[1].header["NAXIS3"]) - frame_3d[1].header["CRPIX3"] + 1
    )

    logger.debug(f"Found {len(axis_pixel)} spaxels along the pixel axis")
    logger.debug(f"Found {len(axis_slice)} spaxels along the slice axis")
    logger.debug(f"Found {len(axis_spectral)} pixels along the spectral axis")

    return axis_pixel, axis_slice, axis_spectral


def determine_target_pixel(
    target_pixel: (tuple[int, int] | list[int] | np.typing.NDArray[np.int_]) = (-1, -1),
    target_radec: (tuple[float, float] | list[float] | np.typing.NDArray[np.float64]) =
                  (0.0, 0.0),
    frame_3d_wcs: WCS = None,
    image: (np.typing.NDArray[np.float64] | None) = None,
            ) -> np.typing.NDArray[np.int_]:

    """
    Determine the target pixel coordinates.

    :param target_pixel: The target pixel coordinates. Default is (-1, -1).
    :param target_radec: The target Right Ascension (RA) and Declination (Dec)
                         coordinates. Default is (0.0, 0.0).
    :param frame_3d_wcs: The World Coordinate System (WCS) object for the 3D frame.
                         Default is None.
    :param image: The image array from which to determine the target pixel if neither
                  target_pixel nor target_radec is provided. Default is None.
    :return:  The determined target pixel coordinates as an array of integers.

    """

    if target_pixel != (-1, -1):
        logger.info("Using provided target pixel provided.")
        target_pixel = target_pixel
    elif target_radec != (0.0, 0.0):
        logger.info(
            "Target Ra and Dec provided. Finding the target pixel from the WCS..."
        )
        target_pixel_3d = frame_3d_wcs.world_to_pixel(
            SkyCoord(
                ra=target_radec[0] * u.deg,
                dec=target_radec[1] * u.deg,
                frame="icrs",
            ),
            2 * u.micron,
        )
        target_pixel = np.asarray(
            [
                (np.round(target_pixel_3d[0]) + 1).astype(int),
                (np.round(target_pixel_3d[1]) + 1).astype(int),
            ]
        )

    elif image is not None:
        logger.warning(
            "Neither target pixel nor target RA and Dec not provided. Finding the"
            "target pixel from the peak of the median image..."
        )
        target_pixel = (
            np.asarray(np.where(image[:, :] == np.nanmax(image[:, :])))[:, 0][::-1] + 1
        ).astype(int)
    else:
        raise ValueError(
            "No target pixel, target RA and Dec, or image provided. Cannot determine"
            "the target pixel."
        )
    target_pixel = np.asarray(target_pixel).astype(int)
    logger.info(f"Using the target pixel {target_pixel} (slice, pixel)")

    return target_pixel


def determine_reference_spectrum(
    frame_3d_shortname,
    axis_spectral,
    frame_3d,
    spaxel_distance,
    inner_radius,
    outer_radius,
    smooth_model_poly_order,
    reference_spectra_percentile_cut=1,
    sigma_clipping_iterations=3,
    sigma_clipping_threshold=5,
):

    logger.debug(f"Finding reference spectra from {frame_3d_shortname}")

    cube_for_computing_reference_spectrum = frame_3d[1].data[...]
    cube_for_computing_reference_spectrum[(frame_3d[3].data[...] != 0)] = np.nan

    # Find the reference spectrum by summing over the central region
    reference_spectrum_indices = np.where(
        (spaxel_distance <= outer_radius) * (spaxel_distance >= inner_radius)
    )

    reference_spectrum = np.nanmean(
        cube_for_computing_reference_spectrum[
            :, reference_spectrum_indices[1], reference_spectrum_indices[0]
        ],
        axis=-1,
    )

    logger.info(
        f'Using {len(reference_spectrum_indices[0])} '
        f'spaxels between {inner_radius}-{outer_radius} '
        f'to compute the reference spectrum'
    )

    # Make a smoothed version of the reference spectrum by fitting with a low-order
    # polynomial
    logger.info(
        f"Fitting {smooth_model_poly_order}-th order "
        f"polynomials to the reference spectra"
    )

    good_pixels = (
        np.isfinite(reference_spectrum)
        & (
            reference_spectrum
            > np.nanpercentile(reference_spectrum, reference_spectra_percentile_cut)
        )
        & (
            reference_spectrum
            < np.nanpercentile(
                reference_spectrum, 100.0 - reference_spectra_percentile_cut
            )
        )
    )

    for j in range(sigma_clipping_iterations + 1):
        pf = np.polyfit(
            axis_spectral[good_pixels],
            reference_spectrum[good_pixels],
            smooth_model_poly_order,
        )

        logger.debug(
            f"The polynomial coefficients for the reference spectrum spectrum are {pf}"
        )
        smoothed_reference_spectrum = np.polyval(pf, axis_spectral)
        rms_difference = np.sqrt(
            np.nanmean(
                (
                    reference_spectrum[good_pixels]
                    - smoothed_reference_spectrum[good_pixels]
                )
                ** 2
            )
        )
        other_bad_pixels = np.abs(reference_spectrum - smoothed_reference_spectrum) > (
            sigma_clipping_threshold * rms_difference
        )
        good_pixels = good_pixels & ~other_bad_pixels

    return reference_spectrum, pf, smoothed_reference_spectrum


def determine_spatial_arr(axis_pixel, axis_slice):

    # Create a meshgrid of the spatial axes
    # axis_pixel.shape = (NAXIS1,)
    # axis_slice.shape = (NAXIS2,)
    # x1.shape = (NAXIS2, NAXIS1)
    # x2.shape = (NAXIS2, NAXIS1)
    # x1[const, :] == axis_pixel
    # x2[:, const] == axis_slice
    x1, x2 = np.meshgrid(axis_pixel, axis_slice)

    # spatial_arr corresponds to locations in the spatial part of the 2D frame
    # spatial_arr[0,...] corresponds to the value along axis_pixel (the pixel axis)
    # spatial_arr[1,...] corresponds to the value along axis_slice (the slice axis)
    # the shape is (2, NAXIS1, NAXIS2)
    spatial_arr = np.asarray([x1.T, x2.T])

    return spatial_arr


def determine_image(frame_3d):

    # We want to plot quick diagnostics of the spectrally collapsed image
    # We'll make a median image
    median_image = np.nanmedian(frame_3d[1].data, axis=0)

    return median_image


def determine_spaxel_distances(spatial_arr, target_pixel):

    spaxel_distance = 3600 * np.sqrt(
        (spatial_arr[0, :, :] - spatial_arr[0, target_pixel[1], target_pixel[0]]) ** 2
        + (spatial_arr[1, :, :] - spatial_arr[1, target_pixel[1], target_pixel[0]]) ** 2
    )

    return spaxel_distance


def initialize_plot(plot: object) -> object:
    """
    We'll save the plots to a pdfpages document
    Either `plot` is a string, in which case we'll save the plots to that file
    Or `plot` is a PdfPages object, in which case we'll append the plots to that object
    If `plot` is False, we won't save the plots
    :param plot:
    :return:
    """
    if type(plot) is str:
        logger.info(f"Saving plots to {plot}")
        plot = PdfPages(plot)
        save_plot = True
    elif plot is not None and plot is not False:
        logger.info("Appending plots to provided pdf")
    else:
        logger.info("Not saving plots")
        plot = False

    return plot


def get_plot_status(plot: object) -> bool:
    """
    Get the status of the plot
    :param plot:
    :return:
    """
    return bool(plot)


def close_plot(plot: object) -> None:
    """
    Close the plot
    :param plot:
    :return:
    """
    if not plot:
        return
    plot.close()


def _build_image_map(
    fig,
    axes,
    axis_pixel: np.ndarray,
    axis_slice: np.ndarray,
    image: np.ndarray,
    spaxel_distance: np.ndarray,
    target_pixel: tuple[int, int],
    frame_3d_shortname: str,
    dewiggling_radius: float,
    central_reference_spectrum_radius: float,
    background_reference_spectrum_radius: tuple[float, float],
):

    horizontal_axis = 3600 * (axis_slice - axis_slice[target_pixel[0]])
    vertical_axis = 3600 * (axis_pixel - axis_pixel[target_pixel[1]])

    axes.pcolormesh(
        horizontal_axis,
        vertical_axis,
        np.log10(image).T,
        cmap="binary_r",
        vmax=np.nanmax(np.log10(image)),
        vmin=np.nanpercentile(np.log10(image), 0.1),
        edgecolors="face",
    )
    axes.contour(
        horizontal_axis,
        vertical_axis,
        spaxel_distance,
        np.sort([dewiggling_radius, central_reference_spectrum_radius]),
        colors="r",
    )
    axes.contour(
        horizontal_axis,
        vertical_axis,
        spaxel_distance,
        np.sort(background_reference_spectrum_radius),
        colors="g",
    )
    axes.scatter(0, 0, c="r", s=100)
    axes.set_aspect("equal")
    axes.set_ylabel("Index 1, Pixel Axis, arcsec")
    axes.set_xlabel("Index 2, Slice Axis, arcsec")
    axes.grid(c="r")
    axes.set_title(
        f"Median image of {frame_3d_shortname}\n Horizontal lines "
        f"have constant pixel number \n Vertical lines have constant slice number"
    )

    return (fig, axes)


def build_image_map(
    plot: object,
    axis_pixel: np.ndarray,
    axis_slice: np.ndarray,
    image: np.ndarray,
    spaxel_distance: np.ndarray,
    target_pixel: tuple[int, int],
    frame_3d_shortname: str,
    dewiggling_radius: float,
    central_reference_spectrum_radius: float,
    background_reference_spectrum_radius: tuple[float, float],
) -> object:
    if not plot:
        return None

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    _build_image_map(
        fig,
        axes,
        axis_pixel,
        axis_slice,
        image,
        spaxel_distance,
        target_pixel,
        frame_3d_shortname,
        dewiggling_radius,
        central_reference_spectrum_radius,
        background_reference_spectrum_radius,
    )

    plot.savefig()
    plt.close()

    return None


def build_reference_spectra_plot(
    plot: object,
    axis_spectral: np.ndarray,
    central_reference_spectrum: np.ndarray,
    smoothed_reference_spectrum: np.ndarray,
    background_reference_spectrum: np.ndarray,
    smoothed_background_reference_spectrum: np.ndarray,
    frame_3d_shortname: str,
    central_reference_spectrum_radius: float,
    background_reference_spectrum_radius: tuple[float, float],
    smooth_model_poly_order: int,
) -> None:
    if not plot:
        return

    logger.info("Plotting the reference spectra")

    fig, axes = plt.subplots(3, 1, figsize=(6, 6))
    fig.suptitle(f"Reference Spectra from {frame_3d_shortname}")
    axes[0].plot(
        axis_spectral[np.isfinite(central_reference_spectrum)],
        central_reference_spectrum[np.isfinite(central_reference_spectrum)],
        c="k",
        label=f'Central Reference Spectrum, mean of central '
              f'{central_reference_spectrum_radius:.2f}"',
    )
    axes[0].plot(
        axis_spectral[np.isfinite(central_reference_spectrum)],
        smoothed_reference_spectrum[np.isfinite(central_reference_spectrum)],
        c="r",
        label=f"Smoothed with order {smooth_model_poly_order} polynomial",
    )
    axes[0].legend(fontsize=6)
    axes[0].set_xlabel("Wavelength, microns")
    axes[0].set_ylabel("Central\n(Flux, arbitrary)")

    axes[1].plot(
        axis_spectral[np.isfinite(background_reference_spectrum)],
        background_reference_spectrum[np.isfinite(background_reference_spectrum)],
        c="k",
        label=(
            f'Background Reference Spectrum, mean between '
            f'({background_reference_spectrum_radius[0]:.2f}'
            f'-{background_reference_spectrum_radius[1]:.2f})"'
        ),
    )
    axes[1].plot(
        axis_spectral[np.isfinite(background_reference_spectrum)],
        smoothed_background_reference_spectrum[
            np.isfinite(background_reference_spectrum)
        ],
        c="r",
        label=f"Smoothed with order {smooth_model_poly_order} polynomial",
    )
    axes[1].legend(fontsize=6)
    axes[1].set_xlabel("Wavelength, microns")
    axes[1].set_ylabel("Background\n(Flux, arbitrary)")

    axes[2].plot(
        axis_spectral[
            np.isfinite(background_reference_spectrum)
            * np.isfinite(central_reference_spectrum)
        ],
        (background_reference_spectrum / central_reference_spectrum)[
            np.isfinite(background_reference_spectrum)
            * np.isfinite(central_reference_spectrum)
        ],
        c="k",
        label="Ratio of background and reference spectra",
    )
    axes[2].plot(
        axis_spectral[
            np.isfinite(background_reference_spectrum)
            * np.isfinite(central_reference_spectrum)
        ],
        (smoothed_background_reference_spectrum / smoothed_reference_spectrum)[
            np.isfinite(background_reference_spectrum)
            * np.isfinite(central_reference_spectrum)
        ],
        c="r",
        label=f"Smoothed with order {smooth_model_poly_order} polynomial",
    )
    axes[2].legend(fontsize=6)
    axes[2].set_xlabel("Wavelength, microns")
    axes[2].set_ylabel("Background / Central\n(Flux, arbitrary)")

    fig.subplots_adjust(left=0.15)
    plot.savefig()
    plt.close()

    return None


def build_bar_map_plot(
    plot: object,
    frame_2d: HDUList,
    bars: np.ndarray,
    frame_2d_shortname: str,
    detector: str,
) -> object:
    if not plot:
        return

    logger.info("Plotting the bars")
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    image_for_map = np.log10(frame_2d[1].data)
    axes.imshow(
        image_for_map,
        cmap="binary",
        origin="lower",
        vmin=np.nanpercentile(image_for_map, 0.1),
        vmax=np.nanpercentile(image_for_map, 99.9),
        interpolation="none",
    )
    for b in range(len(bars)):
        CC = ConvexHull(bars[b])
        vertices = list(CC.vertices[:])
        vertices.append(vertices[0])
        axes.plot(
            bars[b][vertices, 1],
            bars[b][vertices, 0],
            lw=1,
            label=f"Slice {b + 1}",
            c="k",
        )
        if detector == "NRS1":
            axes.text(
                np.min(bars[b][vertices, 1]),
                np.max(bars[b][vertices, 0]),
                f"{b + 1}   ",
                fontsize=8,
                color="k",
                ha="right",
                va="top",
            )
        elif detector == "NRS2":
            axes.text(
                np.max(bars[b][vertices, 1]),
                np.max(bars[b][vertices, 0]),
                f"    {b + 1}",
                fontsize=8,
                color="k",
                ha="left",
                va="top",
            )
    axes.set_ylabel("x pixel")
    axes.set_xlabel("y pixel")
    axes.set_title(f"{frame_2d_shortname}", fontsize=8)
    plot.savefig()
    plt.close()

    return None


def build_spaxel_map_plot(
    plot: object,
    axis_pixel: np.ndarray,
    axis_slice: np.ndarray,
    axis_spectral: np.ndarray,
    image: np.ndarray,
    spaxel_distance: np.ndarray,
    target_pixel: tuple[int, int],
    target_index,
    target_slices: list[int],
    frame_2d: HDUList,
    bars: np.ndarray,
    trace_models: list[
        list[
        Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]
        ]
    ],
    dewiggling_radius: float,
    central_reference_spectrum_radius: float,
) -> None:
    if not plot:
        return None

    logger.debug("Plotting the map and slice traces for the target pixel")

    target_panel_width = 3.0
    right_margin = 0.25
    left_margin = 0.75
    target_figure_width = target_panel_width + left_margin + right_margin

    target_panel_height = 3.0
    gap = 0.75
    bottom_margin = 0.5
    top_margin = 0.5
    target_figure_height = (
        len(target_slices) * target_panel_height
        + (len(target_slices) - 1) * target_panel_height * gap
        + bottom_margin
        + top_margin
    )

    # fig, axes = plt.subplots(1 + len(target_slices), 1, figsize=(4, 4 * (1 + len(target_slices))))
    fig, axes = plt.subplots(
        1 + len(target_slices),
        1,
        figsize=(target_figure_width, target_figure_height),
    )
    fig.subplots_adjust(
        left=left_margin / target_figure_width,
        right=1 - right_margin / target_figure_width,
        bottom=bottom_margin / target_figure_height,
        top=1 - top_margin / target_figure_height,
        hspace=gap / target_panel_height,
    )

    axes[0].pcolormesh(
        3600 * (axis_slice - axis_slice[target_pixel[0]]),
        3600 * (axis_pixel - axis_pixel[target_pixel[1]]),
        np.log10(image).T,
        cmap="binary_r",
        vmax=np.nanmax(np.log10(image)),
        vmin=np.nanpercentile(np.log10(image), 0.1),
        edgecolors="none",
    )
    axes[0].contour(
        3600 * (axis_slice - axis_slice[target_pixel[0]]),
        3600 * (axis_pixel - axis_pixel[target_pixel[1]]),
        spaxel_distance,
        np.sort([dewiggling_radius, central_reference_spectrum_radius]),
        colors="r",
    )
    axes[0].scatter(0, 0, c="r", s=64)

    axes[0].scatter(
        3600 * (axis_slice[target_index[0]] - axis_slice[target_pixel[0]]),
        3600 * (axis_pixel[target_index[1]] - axis_pixel[target_pixel[1]]),
        c="xkcd:orange",
    )

    axes[0].set_ylim(
        3600 * (axis_pixel[target_index[1]] - axis_pixel[target_pixel[1]]) - 0.5,
        3600 * (axis_pixel[target_index[1]] - axis_pixel[target_pixel[1]]) + 0.5,
    )
    axes[0].set_xlim(
        3600 * (axis_slice[target_index[0]] - axis_slice[target_pixel[0]]) - 0.5,
        3600 * (axis_slice[target_index[0]] - axis_slice[target_pixel[0]]) + 0.5,
    )

    axes[0].set_aspect("equal")
    axes[0].set_ylabel("Index 1, Pixel Axis, arcsec")
    axes[0].set_xlabel("Index 2, Slice Axis, arcsec")
    axes[0].grid(c="r")

    image_for_map = np.log10(frame_2d[1].data)

    for tindex, ts in enumerate(target_slices):
        axes[tindex + 1].imshow(
            image_for_map,
            cmap="binary",
            origin="lower",
            vmin=np.nanpercentile(image_for_map, 10.0),
            vmax=np.nanpercentile(image_for_map, 98),
            interpolation="nearest",
        )

        axes[tindex + 1].plot(
            trace_models[tindex][1](axis_spectral),
            trace_models[tindex][0](axis_spectral),
            c="xkcd:orange",
            linestyle=":",
            lw=2,
        )

        axes[tindex + 1].set_ylabel("x pixel")
        axes[tindex + 1].set_xlabel("y pixel")
        axes[tindex + 1].set_xlim(np.min(bars[ts][:, 1]), np.max(bars[ts][:, 1]))
        axes[tindex + 1].set_ylim(np.min(bars[ts][:, 0]), np.max(bars[ts][:, 0]))
        axes[tindex + 1].set_aspect("auto")
        axes[tindex + 1].set_title(f"Slice {ts + 1}")

    plot.savefig()
    plt.close()

    return None


def build_population_diagnostics(
    plot: object,
    axis_pixel: np.ndarray,
    axis_slice: np.ndarray,
    image: np.ndarray,
    spaxel_distance: np.ndarray,
    target_pixel: tuple[int, int],
    frame_3d_shortname: str,
    dewiggling_radius: float,
    central_reference_spectrum_radius: float,
    background_reference_spectrum_radius: tuple[float, float],
    xsol,
    wiggly_pixels,
    diagnostic_fraction=1,
) -> object:

    if not plot:
        return

    target_panel_width = 3.0
    right_margin = 0.25
    left_margin = 0.75
    target_figure_width = target_panel_width + left_margin + right_margin

    target_panel_height = 3.0
    gap = 0.75
    bottom_margin = 0.5
    top_margin = 0.5
    target_figure_height = (
        target_panel_height + target_panel_height * gap + bottom_margin + top_margin
    )

    parsed_xsol = []
    for xx in xsol:
        alpha_order = xx[1]["alpha_order"]
        beta_order = xx[1]["beta_order"]
        gamma_order = xx[1]["gamma_order"]
        order = xx[1]["order"]

        alpha_args = xx[0][0 : alpha_order + 1]
        gamma_args = xx[0][alpha_order + 1 : alpha_order + gamma_order + 2]
        cP = xx[0][alpha_order + beta_order + gamma_order + 3]
        cG = xx[0][alpha_order + beta_order + gamma_order + 4]

        beta_args, wiggle_args, f = parse_wiggle_args(
            xx[0],
            {
                'alpha_order': alpha_order,
                'beta_order': beta_order,
                'gamma_order': gamma_order,
                'order': order,
            },
        )

        wiggle_args = list(wiggle_args)
        while len(wiggle_args) < 10:
            wiggle_args.append(np.nan)
        wiggle_args = np.asarray(wiggle_args)

        alpha_args = list(alpha_args)
        while len(alpha_args) < 10:
            alpha_args.append(np.nan)
        alpha_args = np.asarray(alpha_args)

        gamma_args = list(gamma_args)
        while len(gamma_args) < 10:
            gamma_args.append(np.nan)
        gamma_args = np.asarray(gamma_args)

        beta_args = list(beta_args)
        while len(beta_args) < 10:
            beta_args.append(np.nan)
        beta_args = np.asarray(beta_args)

        parsed_xsol.append([alpha_args, gamma_args, cP, cG, f, beta_args, wiggle_args])

    quantities_to_plot = [
        [xx[0][0] for xx in parsed_xsol],
        [xx[1][0] for xx in parsed_xsol],
        # [xx[2] for xx in parsed_xsol],
        # [xx[3] for xx in parsed_xsol],
        # [xx[0][0] * xx[2] for xx in parsed_xsol],
        # [xx[1][0] * xx[3] for xx in parsed_xsol],
        [xx[4] for xx in parsed_xsol],
        # [xx[5][0] for xx in parsed_xsol],
        [xx[5][1] for xx in parsed_xsol],
        # [xx[5][2] for xx in parsed_xsol],
        # [xx[5][3] for xx in parsed_xsol],
        # [(xx[0][0] * xx[2] + xx[1][0] * xx[3]) for xx in parsed_xsol],
        [xx[6][0] for xx in parsed_xsol],
        [xx[6][1] for xx in parsed_xsol],
        [
            (xx[6][0] ** 2 + xx[6][1] ** 2) ** 0.5
            * (xx[0][0] * xx[2] + xx[1][0] * xx[3])
            for xx in parsed_xsol
        ],
        [xx[6][2] for xx in parsed_xsol],
        [xx[6][3] for xx in parsed_xsol],
        [
            (xx[6][2] ** 2 + xx[6][3] ** 2) ** 0.5
            * (xx[0][0] * xx[2] + xx[1][0] * xx[3])
            for xx in parsed_xsol
        ],
        [xx[6][4] for xx in parsed_xsol],
        [xx[6][5] for xx in parsed_xsol],
        [
            (xx[6][4] ** 2 + xx[6][5] ** 2) ** 0.5
            * (xx[0][0] * xx[2] + xx[1][0] * xx[3])
            for xx in parsed_xsol
        ],
    ]

    labels = [
        "alpha_0",
        "gamma_0",
        # "c_P",
        # "c_G",
        # "c_P alpha_0",
        # "c_G gamma_0",
        "f",
        # "beta_0",
        "beta_1",
        # "beta_2",
        # "beta_3",
        # "(c_P alpha_0 + c_G gamma_0)",
        "a_1",
        "b_1",
        "(c_P alpha_0 + c_G gamma_0) (a_1^2 + b_1^2)^0.5",
        "a_2",
        "b_2",
        "(c_P alpha_0 + c_G gamma_0)  (a_2^2 + b_2^2)^0.5",
        "a_3",
        "b_3",
        "(c_P alpha_0 + c_G gamma_0) (a_3^2 + b_3^2)^0.5",
    ]

    for q, l in zip(quantities_to_plot, labels):

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(target_figure_width, target_figure_height),
        )
        fig.subplots_adjust(
            left=left_margin / target_figure_width,
            right=1 - right_margin / target_figure_width,
            bottom=bottom_margin / target_figure_height,
            top=1 - top_margin / target_figure_height,
            hspace=gap / target_panel_height,
        )

        _build_image_map(
            fig,
            axes[0],
            axis_pixel,
            axis_slice,
            image,
            spaxel_distance,
            target_pixel,
            frame_3d_shortname,
            dewiggling_radius,
            central_reference_spectrum_radius,
            background_reference_spectrum_radius,
        )

        axes[0].set_xlim(-1.5 * dewiggling_radius, 1.5 * dewiggling_radius)
        axes[0].set_ylim(-1.5 * dewiggling_radius, 1.5 * dewiggling_radius)

        overlay_image = np.ones_like(image) * np.nan

        horizontal_axis = 3600 * (axis_slice - axis_slice[target_pixel[0]])
        vertical_axis = 3600 * (axis_pixel - axis_pixel[target_pixel[1]])

        ## Okay, so first, let's plot out alpha_0
        for windex in range(len(q)):
            overlay_image[
                wiggly_pixels[0][::diagnostic_fraction][windex],
                wiggly_pixels[1][::diagnostic_fraction][windex],
            ] = q[windex]

        axes[0].pcolormesh(
            horizontal_axis,
            vertical_axis,
            overlay_image.T,
            cmap="magma",
            vmax=np.nanmax(overlay_image),
            vmin=np.nanmin(overlay_image),
            edgecolors="none",
        )

        axes[1].scatter(
            spaxel_distance.flatten(),
            (overlay_image.T).flatten(),
            c="k",
        )
        axes[0].set_title(f"{l}")
        axes[1].set_ylabel(f"{l}")
        axes[1].set_xlabel("Distance from target pixel, arcsec")

        plot.savefig()
        plt.close()
    return None


def dewiggler_stack(
    frame_2d_filename: str,
    frame_3d_filename: str,
    output_frame_filename: str,
    target_pixel: tuple[int, int] = (-1, -1),
    plot: object = False,
    dewiggling_radius: float = 0.5,
    central_reference_spectrum_radius: float = 0.2,
    smooth_model_poly_order: int = 5,
    diagnostic_fraction: int = 1,
    background_reference_spectrum_radii: tuple[float, float] = (1.0, 1.5),
    fractional_residual_threshold: float = 0.05,
    fit_param_fractional_residual_improvement_threshold: float = 0.001,
    n_jobs: int = 1,
    perform_dewiggling: bool = True,
    target_radec: tuple[float, float] = (0.0, 0.0),
    use_cached_wcs: bool = True,
    min_fit_params: dict = {
        "alpha_order": 0,
        "beta_order": 1,
        "gamma_order": 0,
        "order": 1,
    },
    max_fit_params: dict = {
        "alpha_order": 0,
        "beta_order": 4,
        "gamma_order": 0,
        "order": 5,
    },
    target_wiggle_pixels: list = [],
    mask_bad_fits: bool = True,
    slices_to_use: list = [0, -1, 1],
    reference_spectra_percentile_cut: float = 1,
    try_different_slice_threshold: float = 0.025,
    weight_based_masking_threshold: Optional[float | None] = 0.5,
    fft_order_width=0.1,
    power_ratio_threshold=3.0,
) -> object:
    """
    :type slices_to_use: list
    :param slices_to_use: This is a list of slice numbers to use for the dewiggling. We will estimate the slice number from the pixel location, then incrementally try the slices in this list. A zero here corrsponds to the estimated slice number and a negative number corresponds to one index lower than that estimate.
    :param frame_2d_filename: This is the path to the 2D detector-frame fits file.
    :param frame_3d_filename: This is the path to the 3D rectified fits file.
    :param output_frame_filename: This is the path to the output 3D dewiggled fits file.
    :param target_pixel: This is the central pixel associated with the point-source. If not provided, the code will use either the provided RA and Dec or attempt to find the peak of the median image.
    :param plot: This is either a filename where a resultant pdf file will be placed or a pdfpages object where the plots will be appended.
    :param dewiggling_radius: This is the radius in arcseconds within which the dewiggling will be performed.
    :param central_reference_spectrum_radius: The point-source spectrum will be estimated from an average of pixels within this radius.
    :param smooth_model_poly_order: The order of the polynomial used to smooth the reference spectra.
    :param diagnostic_fraction: For faster diagnostics, only dewiggle every Nth pixel. For example, if diagnostic_fraction=10, only dewiggle every 10th pixel.
    :param background_reference_spectrum_radii: The background spectrum will be estimated from an average of pixels between these radii.
    :param fractional_residual_threshold: If the fractional residual is above this threshold, the fit is considered bad.
    :param fit_param_fractional_residual_improvement_threshold: When iterating through the fit parameters, if the fractional residual improvement is below this threshold, the fit is not considered to be improved.
    :param n_jobs: The number of jobs to use for parallel processing. If plotting is enabled, this is ignored.
    :param perform_dewiggling: Flag whether to actually perform the dewiggling or to just diagnostics.
    :param target_radec: The target Ra and Dec in degrees (ICRS) corresponding to the point source.
    :param use_cached_wcs: Use the cached WCS if available.
    :param min_fit_params: A dictionary containing the minimum fit orders to use. The dictionary should include the keys 'alpha_order', 'beta_order', 'gamma_order', and 'order'.
    :param max_fit_params: A dictionary containing the maximum fit orders to use. The dictionary should include the keys 'alpha_order', 'beta_order', 'gamma_order', and 'order'.
    :param target_wiggle_pixels: A list of specific pixels to dewiggle (e.g., as a diagnostic). If not provided, the code will use all pixels within the dewiggling radius.
    :param mask_bad_fits: If True, replace the spectra in the output cube with NaNs if the fit is bad.
    :param reference_spectra_percentile_cut: When constructing the reference spectra, ignore pixels in the top Nth or bottom Nth percentile. 0 here corresponds to no cut..
    :param weight_based_masking_threshold: Perform a masking of the spectra based on the weights. If the weight is below this fraction of the median weight, the pixel will be masked. 
    :return: The plot object containing all diagnostics.
    """

    logger.info("Initiating dewiggler...")

    frame_2d, frame_3d, frame_2d_shortname, frame_3d_shortname = load_frames(
        frame_2d_filename, frame_3d_filename
    )

    detector = determine_detector_name(frame_2d)

    frame_2d_wcs, frame_3d_wcs = get_wcs(
        frame_2d_filename, frame_3d, use_cached_wcs=use_cached_wcs
    )

    axis_pixel, axis_slice, axis_spectral = get_principal_axes(frame_3d)

    spatial_arr = determine_spatial_arr(axis_pixel, axis_slice)

    plot = initialize_plot(plot)

    logger.info("Constructing a median image")

    image = determine_image(frame_3d)

    target_pixel = determine_target_pixel(
        target_pixel=target_pixel,
        target_radec=target_radec,
        frame_3d_wcs=frame_3d_wcs,
        image=image,
    )

    # TODO need to change all future references to target_pixel to be [y+1, x+1]
    # instead of [x, y]
    target_pixel = (target_pixel[1] - 1, target_pixel[0] - 1)
    target_wiggle_pixels = [[t - 1 for t in tp][::-1] for tp in target_wiggle_pixels]

    spaxel_distance = determine_spaxel_distances(spatial_arr, target_pixel)

    build_image_map(
        plot,
        axis_pixel,
        axis_slice,
        image,
        spaxel_distance,
        target_pixel,
        frame_3d_shortname,
        dewiggling_radius,
        central_reference_spectrum_radius,
        background_reference_spectrum_radii,
    )

    central_reference_spectrum, pf, smoothed_reference_spectrum = (
        determine_reference_spectrum(
            frame_3d_shortname,
            axis_spectral,
            frame_3d,
            spaxel_distance,
            0.0,
            central_reference_spectrum_radius,
            smooth_model_poly_order,
            reference_spectra_percentile_cut=reference_spectra_percentile_cut,
        )
    )

    background_reference_spectrum, pf_back, smoothed_background_reference_spectrum = (
        determine_reference_spectrum(
            frame_3d_shortname,
            axis_spectral,
            frame_3d,
            spaxel_distance,
            background_reference_spectrum_radii[0],
            background_reference_spectrum_radii[1],
            smooth_model_poly_order,
            reference_spectra_percentile_cut=reference_spectra_percentile_cut,
        )
    )

    build_reference_spectra_plot(
        plot,
        axis_spectral,
        central_reference_spectrum,
        smoothed_reference_spectrum,
        background_reference_spectrum,
        smoothed_background_reference_spectrum,
        frame_3d_shortname,
        central_reference_spectrum_radius,
        background_reference_spectrum_radii,
        smooth_model_poly_order,
    )

    logger.debug(f"Finding bar pixels in {frame_2d_shortname}")

    # Now we need to make a map of the bars in the 2D frame
    bars = find_bars(frame_2d[7].data)

    logger.debug(f"Found {len(bars)} bars")
    if len(bars) != 30:
        logger.error(f"Found {len(bars)} bars. This number should be 30...")

    build_bar_map_plot(plot, frame_2d, bars, frame_2d_shortname, detector)

    # Now start iterating through the wiggly pixels
    wiggly_pixels = np.where(
        (~np.isnan(image)) * (spaxel_distance.T < dewiggling_radius)
    )

    if len(target_wiggle_pixels) > 0:
        wiggly_pixels = np.asarray(target_wiggle_pixels).T

    logger.info("Beginning main dewiggling loop...")
    logger.info(
        f'Found {len(wiggly_pixels[0])} wiggly pixels within {dewiggling_radius}"'
    )
    if diagnostic_fraction != 1:
        logger.info(f"Only dewiggling every {diagnostic_fraction}th pixel")

    def dewiggle_main_loop(i: int):

        target_index = [wiggly_pixels[0][i], wiggly_pixels[1][i]]
        dist = spaxel_distance[target_index[1], target_index[0]]

        logger.info("")
        logger.info(
            80 * "~"
        )
        logger.info(
            80 * "~"
        )

        inner_log_msg = (
            f"Begin Dewiggling of spaxel {i+1:6.0f} of {len(wiggly_pixels[0]):6.0f}"
        )
        start_gap = (70 - len(inner_log_msg)) // 2
        end_gap = 70 - ((70 - len(inner_log_msg)) // 2) - len(inner_log_msg)
        logger.info(f'~~~~~{" " * start_gap}{inner_log_msg}{" " * end_gap}~~~~~')

        second_line_msg = f'({target_index[0]}, {target_index[1]}), R = {dist:.2f}"'
        start_gap = (70 - len(second_line_msg)) // 2
        end_gap = 70 - ((70 - len(second_line_msg)) // 2) - len(second_line_msg)
        logger.info(f'~~~~~{" " * start_gap}{second_line_msg}{" " * end_gap}~~~~~')

        logger.info(
            80 * "~"
        )
        logger.info(
            80 * "~"
        )
        logger.info("")

        fp = infer_fractional_position(image, target_index)

        base_target_slice = int(np.floor(fp[0] * len(bars)))

        target_slices = [base_target_slice + s for s in slices_to_use]

        logger.debug(
            f"This spaxel is estimated to be in slice {base_target_slice + 1} "
            f"(fractional position {fp[0]:.2f})"
        )

        logger.debug(
            f"For this spaxel, we will try fitting with slices"
            f"{[t + 1 for t in target_slices]}"
        )

        if weight_based_masking_threshold is not None:
            weight_spectrum = frame_3d[4].data[:, target_index[0], target_index[1]]
            median_weight = np.nanmedian(weight_spectrum[weight_spectrum > 0.0])
            weight_based_mask = (
                weight_spectrum > weight_based_masking_threshold * median_weight
            )

        else:
            logger.debug("Weight-based masking is disabled.")
            weight_based_mask = None
        



        # find world coordinates for the target pixel
        world_coords = frame_3d_wcs.pixel_to_world(
            target_index[1], target_index[0], 1000
        )

        trace_models = [
            get_trace_model(
                frame_2d_wcs[slice_index],
                world_coords[0].ra.deg,
                world_coords[0].dec.deg,
            )
            for slice_index in target_slices
        ]

        build_spaxel_map_plot(
            plot,
            axis_pixel,
            axis_slice,
            axis_spectral,
            image,
            spaxel_distance,
            target_pixel,
            target_index,
            target_slices,
            frame_2d,
            bars,
            trace_models,
            dewiggling_radius,
            central_reference_spectrum_radius,
        )

        # For good measure, we'll try the de-wiggling on the slice before and after the target slice and use the best model of the three.

        base_x2 = 1e10
        preferred_wiggle_model = np.ones_like(axis_spectral) * np.nan
        preferred_dewiggled_spectrum = np.ones_like(axis_spectral) * np.nan
        preferred_wiggle_fft = {}

        logger.debug(f"Starting the dewiggler for index {i}!")
        for tindex, target_slice in enumerate(target_slices):
            if base_x2**0.5 < try_different_slice_threshold:
                continue

            logger.info(
                "-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~"
            )
            logger.info(
                f"  Current √𝜈 = {base_x2**0.5:.4f} > {try_different_slice_threshold:.4f}"
            )
            logger.info(f"  Trying trace from slice {target_slice + 1}")
            logger.info(
                "-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~"
            )

            model_trace = [
                [trace_models[tindex][0], trace_models[tindex][0]],
                [trace_models[tindex][0], trace_models[tindex][0]],
            ]

            wiggle_model, dewiggled_spectrum, x2, xsol, model_details, wiggle_fft = (
                dewiggle_lambda_space(
                    model_trace,
                    pf,
                    axis_spectral,
                    frame_3d[1].data[:, target_index[0], target_index[1]],
                    frame_3d[3].data[:, target_index[0], target_index[1]],
                    plot=plot,
                    plot_label=f'({target_index[0]},{target_index[1]}), r = {dist:.3f}", Slice {target_slice + 1}',
                    background_pf=pf_back,
                    min_fit_params=min_fit_params,
                    max_fit_params=max_fit_params,
                    sqrt_nu_improvement_threshold=(
                        fit_param_fractional_residual_improvement_threshold
                    ),
                    target_maskarray=weight_based_mask,
                )
            )

            logger.info(
                40 * "-~"
            )
            logger.info(f"  Found √𝜈 = {x2**0.5:.4f} using slice {target_slice + 1}")
            logger.info(f"  Previous best was {base_x2**0.5:.4f}")
            logger.info(
                40 * "-~"
            )

            if (x2 < base_x2) or (preferred_wiggle_model is None):
                base_x2 = x2
                preferred_wiggle_model = wiggle_model
                preferred_dewiggled_spectrum = dewiggled_spectrum
                preferred_wiggle_fft = wiggle_fft

        return (
            xsol,
            model_details,
            target_index,
            preferred_dewiggled_spectrum,
            preferred_wiggle_fft,
            base_x2,
        )

    if get_plot_status(plot):
        logger.warning(
            "Plotting is active, so defaulting to serial processing with n_jobs=1"
        )
        n_jobs = 1

    xsol_fn = output_frame_filename.replace(".fits", ".pkl")

    if perform_dewiggling:

        total_steps = len(range(len(wiggly_pixels[0]))[::diagnostic_fraction])
        xsols = ProgressParallel(
            n_jobs=n_jobs, verbose=0, require="sharedmem", total=total_steps
        )(
            delayed(dewiggle_main_loop)(i)
            for i in range(len(wiggly_pixels[0]))[::diagnostic_fraction]
        )
        d.dump(xsols, open(xsol_fn, "wb"))

    xsols = d.load(open(xsol_fn, "rb"))

    for xindex, x in enumerate(xsols):
    
        bad_fit = False

        retries = 0
        retry_allowed = True

        while retry_allowed:

            if retries == 1:
                retry_allowed = False

            retries += 1

            target_index = x[2]
            preferred_dewiggled_spectrum = x[3]
            fit_x2 = x[5]

            if fit_x2**0.5 > fractional_residual_threshold:
                bad_fit = True

            # Compute the amplitudes of the fft modes.

            fft_freq = x[4]['freq']
            p1_indices = (fft_freq[:] >= 1 - fft_order_width) & (fft_freq[:] <= 1 + fft_order_width)
            # p32_indices = (fft_freq[:] >= 1.5 - fft_order_width) & (fft_freq[:] <= 1.5 + fft_order_width)

            p1 = np.mean(np.abs(x[4]['residual_fft'][p1_indices]))
            pavg = np.mean(np.abs(x[4]['residual_fft']))
            logger.info('')

            if p1 / pavg > power_ratio_threshold:
                bad_fit = True

            logger.info(f"Pixel ({target_index[0]}, {target_index[1]}) has √𝜈 = {fit_x2**0.5:.4f}."
                        f"    and P(1) / P(3/2) = {p1 / pavg:.4f}.")

            if mask_bad_fits & bad_fit:
                logger.warning("Masking this pixel.")
                
                if retry_allowed:
                    frame_3d[1].data[:, target_index[0], target_index[1]] = preferred_dewiggled_spectrum
                    x = dewiggle_main_loop(xindex)

                else:
                    preferred_dewiggled_spectrum = np.ones_like(preferred_dewiggled_spectrum) * np.nan

        
        frame_3d[1].data[
            :, target_index[0], target_index[1]
        ] = preferred_dewiggled_spectrum

    logger.info("Dewiggling complete!")

    build_population_diagnostics(
        plot,
        axis_pixel,
        axis_slice,
        image,
        spaxel_distance,
        target_pixel,
        frame_3d_shortname,
        dewiggling_radius,
        central_reference_spectrum_radius,
        background_reference_spectrum_radii,
        xsols,
        wiggly_pixels,
        diagnostic_fraction=diagnostic_fraction,
    )
    close_plot(plot)

    frame_3d.writeto(output_frame_filename, overwrite=True)

    return plot


def parse_yaml_arguments_and_dewiggle(getter):

    input_2d_prototype = getter("input_2d_filename_prototype")
    input_3d_prototype = getter("input_3d_filename_prototype")
    output_filename_prototype = getter("output_filename_prototype")
    target_detectors = getter("target_detectors")
    target_frames = getter("target_frames")

    assert target_frames is not [
        ""
    ], "No target frames specified and I don't yet know how to handle that..."

    log_filename_prototype = output_filename_prototype.replace(".fits", ".log")
    plot_filename_prototype = getter("plot_filename_prototype")

    for target_frame in target_frames:
        for target_detector in target_detectors:

            log_filename = log_filename_prototype.replace(
                "DETECTOR", target_detector
            ).replace("FRAME", target_frame)

            if plot_filename_prototype:
                plot_filename = plot_filename_prototype.replace(
                    "DETECTOR", target_detector
                ).replace("FRAME", target_frame)
            else:
                plot_filename = False

            # logger = logging.getLogger()

            with open(log_filename, "w"):
                pass

            logging.basicConfig(
                format="%(asctime)s: %(funcName)-24s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %I:%M:%S %p",
                filename=log_filename,
                level=logging.INFO,
                force=True,
            )

            logger.info(
                f'Beginning dewiggler task {getter("label", "")} on '
                f'{input_3d_prototype.replace("DETECTOR", target_detector)
                    .replace("FRAME", target_frame).split("/")[-1]}'
            )

            dewiggler_stack(
                input_2d_prototype.replace("DETECTOR", target_detector).replace(
                    "FRAME", target_frame
                ),
                input_3d_prototype.replace("DETECTOR", target_detector).replace(
                    "FRAME", target_frame
                ),
                output_filename_prototype.replace("DETECTOR", target_detector).replace(
                    "FRAME", target_frame
                ),
                target_pixel=getter("central_pixel"),
                plot=plot_filename,
                dewiggling_radius=getter("dewiggling_radius"),
                central_reference_spectrum_radius=getter(
                    "central_reference_spectrum_radius"
                ),
                diagnostic_fraction=getter("diagnostic_fraction"),
                smooth_model_poly_order=getter("smooth_model_poly_order"),
                background_reference_spectrum_radii=getter(
                    "background_reference_spectrum_radii"
                ),
                fractional_residual_threshold=getter("fractional_residual_threshold"),
                fit_param_fractional_residual_improvement_threshold=getter(
                    "fit_param_fractional_residual_improvement_threshold"
                ),
                n_jobs=int(getter("n_jobs")),
                perform_dewiggling=bool(getter("perform_dewiggling")),
                target_radec=tuple(getter("target_radec")),
                use_cached_wcs=bool(getter("use_cached_wcs")),
                min_fit_params=getter("min_fit_params"),
                max_fit_params=getter("max_fit_params"),
                target_wiggle_pixels=getter("target_wiggly_pixels"),
                mask_bad_fits=bool(getter("mask_bad_fits")),
                slices_to_use=getter("slices_to_use"),
                reference_spectra_percentile_cut=getter(
                    "reference_spectra_percentile_cut"
                ),
                try_different_slice_threshold=getter("try_different_slice_threshold"),
            )

    return 0


default_options = {
    "input_2d_filename_prototype": None,
    "input_3d_filename_prototype": None,
    "output_filename_prototype": None,
    "target_detectors": ["nrs1", "nrs2"],
    "target_frames": [""],
    "plot_filename_prototype": False,
    "central_pixel": [-1, -1],
    "dewiggling_radius": 0.5,
    "central_reference_spectrum_radius": 0.2,
    "smooth_model_poly_order": 5,
    "diagnostic_fraction": 1,
    "background_reference_spectrum_radii": [1.0, 1.5],
    "fractional_residual_threshold": 0.05,
    "fit_param_fractional_residual_improvement_threshold": 0.0001,
    "n_jobs": 1,
    "perform_dewiggling": True,
    "target_radec": (0.0, 0.0),
    "use_cached_wcs": True,
    "target_wiggly_pixels": [],
    "mask_bad_fits": True,
    "slices_to_use": [0, -1, 1],
    "reference_spectra_percentile_cut": 1,
    "try_different_slice_threshold": 0.025,
    "min_fit_params": {
        "alpha_order": 0,
        "beta_order": 1,
        "gamma_order": 0,
        "order": 3,
    },
    "max_fit_params": {
        "alpha_order": 0,
        "beta_order": 1,
        "gamma_order": 0,
        "order": 3,
    },
}

necessary_arguments = (
    "label",
    "input_2d_filename_prototype",
    "input_3d_filename_prototype",
    "output_filename_prototype",
    "target_frames",
)

step = {
    "target_step_type": "dewiggle",
    "target_function": parse_yaml_arguments_and_dewiggle,
    "default_options": default_options,
    "necessary_arguments": necessary_arguments,
}


def dewiggler_wrapper(yaml_file):

    ingest_yaml(yaml_file, [step])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dewiggle a frame")
    parser.add_argument(
        "yaml_file",
        type=str,
        help="The yaml file which contains the arguments for the dewiggler",
    )
    args = parser.parse_args()

    dewiggler_wrapper(args.yaml_file)
