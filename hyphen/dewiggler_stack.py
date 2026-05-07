import logging
import argparse

from astropy.io.fits import HDUList
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
from scipy.ndimage import median_filter
from scipy.interpolate import make_interp_spline
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from stdatamodels.jwst.datamodels import IFUImageModel
from stdatamodels.jwst.datamodels import MultiSlitModel
from joblib import delayed
import dill as d
import json
import pickle
from typing import Callable, Optional
from functools import partial
import warnings
import tempfile
import subprocess
import shutil
import os
from jwst import datamodels


class InterpolatedPf:
    """Wraps a B-spline interpolator so it can be used wherever polynomial
    coefficient arrays (``pf``) are expected.

    When ``smooth_model_poly_order == -1`` the reference-spectrum fit is
    replaced by a B-spline over the cleaned spectrum.  Downstream code that
    calls ``np.polyval(pf, x)`` must instead call ``_eval_pf(pf, x)``; code
    that calls ``len(pf)`` to size a background-prototype fit must instead
    call ``_pf_order(pf)``.

    Parameters
    ----------
    spline : callable
        A ``scipy.interpolate.BSpline`` (or any callable ``f(x) -> array``).
    effective_order : int
        The polynomial-equivalent complexity, used to size the
        ``background_prototype_pf`` fit in ``dewiggle_lambda_space``.
    """

    def __init__(self, spline: Callable, effective_order: int) -> None:
        self._spline = spline
        self.effective_order = effective_order

    def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return self._spline(x)

    def __len__(self) -> int:  # keeps any accidental len() calls from hard-crashing
        return self.effective_order + 1

# In-memory WCS cache to avoid repeated loading across batch iterations
_wcs_cache: dict[tuple[str, str], tuple] = {}

np.random.seed(67)

def _save_fit_results(xsols: list, base_filename: str) -> None:
    """Save fit results to compressed npz + json for fast I/O."""
    npz_filename = base_filename + ".npz"
    json_filename = base_filename + ".json"

    n = len(xsols)
    # Use dtype=object because fit escalation means xsol arrays have different
    # lengths across spaxels. FFT arrays can also vary in length.
    xsols_array = np.array([x[0] for x in xsols], dtype=object)
    target_indices = np.array([x[2] for x in xsols])
    dewiggled_spectra = np.array([x[3] for x in xsols])
    base_x2s = np.array([x[5] for x in xsols])
    preferred_target_slices = np.array([x[7] for x in xsols])

    # FFT components
    fft_freqs = np.array([x[4].get("freq", np.array([])) for x in xsols], dtype=object)
    fft_spectra = np.array([x[4].get("spectrum_fft", np.array([])) for x in xsols], dtype=object)
    fft_residuals = np.array([x[4].get("residual_fft", np.array([])) for x in xsols], dtype=object)

    np.savez_compressed(
        npz_filename,
        xsols=xsols_array,
        target_indices=target_indices,
        dewiggled_spectra=dewiggled_spectra,
        base_x2s=base_x2s,
        fft_freqs=fft_freqs,
        fft_spectra=fft_spectra,
        fft_residuals=fft_residuals,
        target_slices=preferred_target_slices,
    )

    # Metadata (dicts, strings) in JSON
    metadata = {
        "n": n,
        "model_details": [x[1] for x in xsols],
        "log_strings": [x[6] for x in xsols],
    }
    with open(json_filename, "w") as f:
        json.dump(metadata, f)


def _load_fit_results(base_filename: str) -> list:
    """Load fit results from compressed npz + json."""
    npz_filename = base_filename + ".npz"
    json_filename = base_filename + ".json"

    data = np.load(npz_filename, allow_pickle=True)
    with open(json_filename, "r") as f:
        metadata = json.load(f)

    n = metadata["n"]
    model_details = metadata["model_details"]
    log_strings = metadata["log_strings"]

    xsols = []
    for i in range(n):
        wiggle_fft = {
            "freq": data["fft_freqs"][i],
            "spectrum_fft": data["fft_spectra"][i],
            "residual_fft": data["fft_residuals"][i],
        }
        xsols.append((
            data["xsols"][i],
            model_details[i],
            data["target_indices"][i].tolist(),
            data["dewiggled_spectra"][i],
            wiggle_fft,
            float(data["base_x2s"][i]),
            log_strings[i],
            data['target_slices'][i]
        ))

    data.close()
    return xsols


def _flush_spaxel_logs(xsols: list) -> None:
    """Write captured per-spaxel fit logs into the root logger's file stream."""
    root_handlers = [
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler)
    ]
    if not root_handlers:
        return
    stream = root_handlers[0].stream
    for x in xsols:
        log_text = x[6]
        if log_text:
            stream.write(log_text)
            if not log_text.endswith("\n"):
                stream.write("\n")
    stream.flush()


warnings.filterwarnings("error", message="Mean of empty slice")
warnings.filterwarnings("error", message="invalid value encountered in scalar divide")

from .tracer_helpers import find_bars, infer_fractional_position
from .dewiggle_lambda_space import dewiggle_lambda_space, parse_wiggle_args, plot_dewiggle_result
from .yaml_helpers import ingest_yaml
from .helpers import ProgressParallel, get_ybounds
from .single_dewiggle_ifu_step import (single_dewiggle_ifu_default_options,
                                       single_dewiggle_ifu_necessary_arguments)
from .batch_dewiggle_ifu_step import (batch_dewiggle_ifu_default_options,
                                      batch_dewiggle_ifu_necessary_arguments)
from .single_dewiggle_msa_step import (single_dewiggle_msa_default_options,
                                       single_dewiggle_msa_necessary_arguments)
from .batch_dewiggle_msa_step import (batch_dewiggle_msa_default_options,
                                      batch_dewiggle_msa_necessary_arguments)


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


def get_wcs_ifu(
    fn_2d: str, fn_3d: str
) -> tuple[WCS, WCS, IFUImageModel, datamodels.IFUCubeModel]:
    """
    Get the wcs for the 2D and 3D frames.

    :param fn_2d: This is the filename for the 2D detector-frame fits file
    :param fn_3d: This is the filename for the 3D rectified fits file
    :return: A tuple containing the single global WCS for the 2D frame, the WCS for
        the 3D frame, and the IFUImageModel (must be kept alive by the caller so the
        underlying ASDF file handle is not garbage-collected).
    """
    cache_key = (fn_2d, fn_3d)
    if cache_key in _wcs_cache:
        logger.info("Using cached WCS")
        return _wcs_cache[cache_key]

    logger.info("Loading global IFU WCS from IFUImageModel")
    ifu_model = IFUImageModel(fn_2d)
    frame_2d_wcs = ifu_model.meta.wcs

    logger.info("Computing WCS for the 3D frame")
    cube_model = datamodels.open(fn_3d)
    frame_3d_wcs = cube_model.meta.wcs

    result = (frame_2d_wcs, frame_3d_wcs, ifu_model, cube_model)
    _wcs_cache[cache_key] = result
    return result


def get_wcs_msa(
    fn_2d: str, fn_3d: str
) -> tuple[
    list[WCS],
    list[WCS],
    MultiSlitModel,
    MultiSlitModel,
]:
    """
    Get the wcs for the 2D and 3D frames

    :param fn_2d: This is the filename for the 2D detector-frame fits file
    :param frame_3d: This is the fits object for the 3D detector-frame fits file.
    :param use_cached_wcs: If True, we'll try to use a cached WCS file. If False, we'll
        compute the WCS from scratch. If there's an issue with loading the cached WCS,
        we'll compute the WCS from scratch.
    :return: A tuple containing a list of the WCS for each slit in the 2D frame and
        the WCS for each slit in the for the 3D frame
    """

    cache_key = (fn_2d, fn_3d)
    if cache_key in _wcs_cache:
        logger.info("Using cached WCS")
        return _wcs_cache[cache_key]

    logger.info("Computing WCS for the slices in the 2D frame")

    frame_2d_model = MultiSlitModel(fn_2d)
    n_slits = len(frame_2d_model.slits)
    frame_2d_wcs = [frame_2d_model.slits[i].meta.wcs for i in range(n_slits)]

    logger.info("Computing WCS for the 3D frame")

    frame_3d_model = MultiSlitModel(fn_3d)
    n_slits = len(frame_3d_model.slits)
    frame_3d_wcs = [frame_3d_model.slits[i].meta.wcs for i in range(n_slits)]

    result = (frame_2d_wcs, frame_3d_wcs, frame_2d_model, frame_3d_model)
    _wcs_cache[cache_key] = result
    return result


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
    detector_wcs: WCS, ra: float, dec: float, mapped_wcs_coord: str = 'detector',
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
        return np.atleast_2d(detector_wcs.get_transform("world", mapped_wcs_coord)(ra, dec, x))

    def trace_x(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return generic_trace(x)[1, :]

    def trace_y(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return generic_trace(x)[0, :]

    return trace_x, trace_y

def get_principal_axes_ifu(
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

def get_principal_axes_msa(
    frame_3d: HDUList,
    frame_3d_wcs: list[WCS],
    slitlet: int,
    ) -> tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
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
    # hdr = frame_3d[8 * slitlet + 1].header
    slitlet_frame = frame_3d[8 * slitlet + 1]
    slitlet_3d_wcs = frame_3d_wcs[slitlet]

    axis_pixel = np.arange(slitlet_frame.data.shape[0])
    axis_slice = np.zeros(1)
    axis_spectral = np.asarray(slitlet_3d_wcs.pixel_to_world_values(np.arange(slitlet_frame.data.shape[1]), np.ones_like(np.arange(slitlet_frame.data.shape[1])) * slitlet_frame.data.shape[0]//2))[2,:]

    logger.debug(f"Found {len(axis_pixel)} spaxels along the pixel axis")
    logger.debug(f"Found {len(axis_spectral)} pixels along the spectral axis")

    return axis_pixel, axis_slice, axis_spectral


def determine_target_pixel_ifu(
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


def determine_target_pixel_msa(
    target_pixel: int = -1,
    target_radec: (tuple[float, float] | list[float] | np.typing.NDArray[np.float64]) =
                  (0.0, 0.0),
    frame_3d_wcs: WCS = None,
    image: (np.typing.NDArray[np.float64] | None) = None,
            ) -> np.typing.NDArray[np.int_]:

    """
    Determine the target pixel coordinates.

    :param target_pixel: The target pixel coordinates. Default is (-1, -1).
                         this is assumed to be 1-indexed; similar to what we'd see in
                         qfitsview or similar. TODO!
    :param target_radec: The target Right Ascension (RA) and Declination (Dec)
                         coordinates. Default is (0.0, 0.0).
    :param frame_3d_wcs: The World Coordinate System (WCS) object for the 3D frame.
                         Default is None.
    :param image: The image array from which to determine the target pixel if neither
                  target_pixel nor target_radec is provided. Default is None.
    :return:  The determined target pixel coordinates as an array of integers.

    """

    if target_pixel != -1:
        logger.info("Using provided target pixel.")
        target_pixel = np.asarray([target_pixel, 1])
    # elif target_radec != (0.0, 0.0):
    #     logger.info(
    #         "Target Ra and Dec provided. Finding the target pixel from the WCS..."
    #     )
    #     target_pixel_3d = frame_3d_wcs.world_to_pixel(
    #         SkyCoord(
    #             ra=target_radec[0] * u.deg,
    #             dec=target_radec[1] * u.deg,
    #             frame="icrs",
    #         ),
    #         2 * u.micron,
    #     )
    #     target_pixel = np.asarray(
    #         [
    #             (np.round(target_pixel_3d[0]) + 1).astype(int),
    #             (np.round(target_pixel_3d[1]) + 1).astype(int),
    #         ]
    #     )

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
    frame_3d_shortname: str,
    axis_spectral: np.typing.NDArray[np.float64],
    frame_3d: HDUList,
    spaxel_distance: np.typing.NDArray[np.float64],
    inner_radius: float,
    outer_radius: float,
    smooth_model_poly_order: int,
    smooth_model_window: int = 51,
    reference_spectra_percentile_cut: float = 1.0,
    sigma_clipping_iterations: int = 3,
    sigma_clipping_threshold: float = 5,
    data_frame_index: int = 1,
    data_quality_frame_index: Optional[int] = 3,
) -> tuple[np.typing.NDArray[np.float64],
           np.typing.NDArray[np.float64] | InterpolatedPf,
           np.typing.NDArray[np.float64]]:

    logger.debug(f"Finding reference spectra from {frame_3d_shortname}")

    cube_for_computing_reference_spectrum = frame_3d[data_frame_index].data[...]

    if data_quality_frame_index is not None:
        invalid_mask = frame_3d[data_quality_frame_index].data[...] != 0
        cube_for_computing_reference_spectrum[invalid_mask] = np.nan

    if cube_for_computing_reference_spectrum.ndim == 2:
        cube_for_computing_reference_spectrum = cube_for_computing_reference_spectrum.T
        cube_for_computing_reference_spectrum = cube_for_computing_reference_spectrum.reshape(
            (
                cube_for_computing_reference_spectrum.shape[0],
                1,
                cube_for_computing_reference_spectrum.shape[1],
            )
        )
    
    if inner_radius < 0 or outer_radius < 0:
        # If either radius is negative, exclude this reference spectrum
        return (np.zeros(cube_for_computing_reference_spectrum.shape[0]),
                np.asarray([0.0]),
                np.zeros(cube_for_computing_reference_spectrum.shape[0]))


    # Find the reference spectrum by summing over the central region
    reference_spectrum_indices = np.where(
        (spaxel_distance <= outer_radius) * (spaxel_distance >= inner_radius)
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        reference_spectrum = np.nanmean(
            cube_for_computing_reference_spectrum[
                :, reference_spectrum_indices[1], reference_spectrum_indices[0]
            ],
            axis=-1,
        )

        for warning in w:
            if "Mean of empty slice" in str(warning.message):
                logger.warning(f"The reference spectrum region includes some empty spaxels")


    # print(f'{reference_spectrum.shape=}')

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

    if sum(good_pixels) == 0:
        pf = [np.nan]
        smoothed_reference_spectrum = np.full_like(reference_spectrum, np.nan)
        return reference_spectrum, pf, smoothed_reference_spectrum

    if smooth_model_poly_order == -1:
        logger.info(
            f"Using median-filter interpolation (window={smooth_model_window}) "
            f"to build the reference spectrum model"
        )
        filtered = median_filter(reference_spectrum, size=smooth_model_window, mode="nearest")
        rms_difference = np.sqrt(
            np.nanmean((reference_spectrum[good_pixels] - filtered[good_pixels]) ** 2)
        )
        for _ in range(sigma_clipping_iterations):
            outlier_mask = np.abs(reference_spectrum - filtered) > (
                sigma_clipping_threshold * rms_difference
            )
            cleaned = reference_spectrum.copy()
            cleaned[outlier_mask] = filtered[outlier_mask]
            filtered = median_filter(cleaned, size=smooth_model_window, mode="nearest")
            rms_difference = np.sqrt(
                np.nanmean(
                    (reference_spectrum[good_pixels & ~outlier_mask]
                     - filtered[good_pixels & ~outlier_mask]) ** 2
                )
            )

        finite_mask = np.isfinite(filtered)
        spline = make_interp_spline(
            axis_spectral[finite_mask],
            filtered[finite_mask],
            k=3,
        )
        pf = InterpolatedPf(spline, effective_order=smooth_model_window // 2)
        smoothed_reference_spectrum = filtered
    else:
        logger.info(
            f"Fitting {smooth_model_poly_order}-th order "
            f"polynomials to the reference spectra"
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


def determine_image(frame_3d, frame_index=1, collapse_axis=0):

    # We want to plot quick diagnostics of the spectrally collapsed image
    # We'll make a median image

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        median_image = np.nanmedian(frame_3d[frame_index].data, axis=collapse_axis)

        logger.info(f"{np.sum(np.isnan(median_image))} of {median_image.size} pixels in the images are all NaN")

    if median_image.ndim == 1:
        median_image = np.atleast_2d(median_image)

    return median_image


def determine_spaxel_distances(spatial_arr, target_pixel, conversion_factor=3600):

    spaxel_distance = conversion_factor * np.sqrt(
        (spatial_arr[0, :, :] - spatial_arr[0, target_pixel[1], target_pixel[0]]) ** 2
        + (spatial_arr[1, :, :] - spatial_arr[1, target_pixel[1], target_pixel[0]]) ** 2
    )

    return spaxel_distance


def initialize_plot(plot: str | PdfPages | None | bool) -> PdfPages | bool:
    """
    We'll save the plots to a pdfpages document
    Either `plot` is a string, in which case we'll save the plots to that file
    Or `plot` is a PdfPages object, in which case we'll append the plots to that object
    If `plot` is False, we won't save the plots
    :param plot:
    :return:
    """
    if isinstance(plot, str):
        logger.info(f"Saving plots to {plot}")
        plot = PdfPages(plot)
    elif isinstance(plot, PdfPages):
        logger.info("Appending plots to provided pdf")
    else:
        logger.info("Not saving plots")
        plot = False

    return plot


def get_plot_status(plot: PdfPages | bool | None) -> bool:
    """
    Get the status of the plot
    :param plot:
    :return:
    """
    return bool(plot)


def close_plot(plot: PdfPages | bool | None) -> None:
    """
    Close the plot
    :param plot:
    :return:
    """
    if not plot:
        return
    if isinstance(plot, PdfPages):
        plot.close()


def _build_image_map_ifu(
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


def _build_image_map_msa(
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

    vertical_axis = (axis_pixel - axis_pixel[target_pixel[1]])

    axes.scatter(vertical_axis, (image)[0, :], c='k')
    
    axes.axvspan(
                -dewiggling_radius,
                dewiggling_radius,
                color='b',
                alpha=0.3, label='Dewiggling Region',
                edgecolor=None)

    axes.axvspan(
                -central_reference_spectrum_radius,
                central_reference_spectrum_radius,
                color='r',
                alpha=0.3, label='Central Spectrum region',
                edgecolor=None)
    axes.axvspan(np.nan, np.nan, color='g',
                 alpha=0.3, label='Background Reference Spectrum Region',
                 edgecolor=None)
        
    if -1 not in background_reference_spectrum_radius:
        for g in [-1,1]:
            axes.axvspan(
                g*background_reference_spectrum_radius[0],
                g*background_reference_spectrum_radius[1],
                color='g',
                alpha=0.3,
                edgecolor=None)

    axes.legend(fontsize=6)
    axes.set_ylabel("Flux\n(arbitrary)")
    axes.set_xlabel("Pixel Axis\n(px; index 1; relative to central pixel)")

    axes.set_title(
        f"Median image of\n{frame_3d_shortname}"
    )

    return (fig, axes)



def build_image_map_ifu(
    plot: PdfPages | bool | None,
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
    _build_image_map_ifu(
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

    if isinstance(plot, PdfPages):
        plot.savefig()
    plt.close()

    return None


def build_image_map_msa(
    plot: PdfPages | bool | None,
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

    fig, axes = plt.subplots(1, 1, figsize=(6, 3.75))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.18)
    
    _build_image_map_msa(
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
    
    if isinstance(plot, PdfPages):
        plot.savefig()
    plt.close()

    return None


def _smooth_model_label(smooth_model_poly_order: int, smooth_model_window: int) -> str:
    if smooth_model_poly_order == -1:
        return f"Median-filter interpolation (window={smooth_model_window})"
    return f"Order {smooth_model_poly_order} polynomial"


def build_reference_spectra_plot(
    plot: PdfPages | bool | None,
    axis_spectral: np.ndarray,
    central_reference_spectrum: np.ndarray,
    smoothed_reference_spectrum: np.ndarray,
    background_reference_spectrum: np.ndarray,
    smoothed_background_reference_spectrum: np.ndarray,
    frame_3d_shortname: str,
    central_reference_spectrum_radius: float,
    background_reference_spectrum_radius: tuple[float, float],
    smooth_model_poly_order: int,
    smooth_model_window: int = 51,
    unit: str = 'arcsec',
    wave_unit: str = 'micron',
    linewidth: float = 0.5,
) -> None:
    if not plot:
        return

    logger.info("Plotting the reference spectra")

    fig, axes = plt.subplots(3, 1, figsize=(6, 9.7))
    fig.suptitle(f"Reference Spectra from \n{frame_3d_shortname}")
    
    axes[0].plot(
        axis_spectral[np.isfinite(central_reference_spectrum)],
        central_reference_spectrum[np.isfinite(central_reference_spectrum)],
        c="k",
        label=f'Central Spectrum\nMean of central '
              f'{central_reference_spectrum_radius:.2f} {unit}',
        linewidth=linewidth,
    )

    yrange = get_ybounds([1e50, 1e-50],
                         central_reference_spectrum[
                            np.isfinite(central_reference_spectrum)
                         ])

    axes[0].plot(
        axis_spectral[np.isfinite(central_reference_spectrum)],
        smoothed_reference_spectrum[np.isfinite(central_reference_spectrum)],
        c="r",
        label=_smooth_model_label(smooth_model_poly_order, smooth_model_window),
        linewidth=2*linewidth,
    )

    yrange = get_ybounds(yrange,
                         smoothed_reference_spectrum[
                            np.isfinite(central_reference_spectrum)
                         ])

    axes[0].legend(fontsize=6)
    axes[0].set_xlabel(f"Wavelength ({wave_unit})")
    axes[0].set_ylabel("Central\n(Flux, arbitrary)")

    axes[0].set_ylim(yrange[0] * 0.9, yrange[1] * 1.1)
    
    axes[1].plot(
        axis_spectral[np.isfinite(background_reference_spectrum)],
        background_reference_spectrum[np.isfinite(background_reference_spectrum)],
        c="k",
        label=(
            f'Background Spectrum\nMean between '
            f'({background_reference_spectrum_radius[0]:.2f}'
            f'-{background_reference_spectrum_radius[1]:.2f}) {unit}'
        ),
        linewidth=linewidth,
    )

    yrange = get_ybounds([1e50, 1e-50],
                         background_reference_spectrum[
                            np.isfinite(background_reference_spectrum)
                        ])

    axes[1].plot(
        axis_spectral[np.isfinite(background_reference_spectrum)],
        smoothed_background_reference_spectrum[
            np.isfinite(background_reference_spectrum)
        ],
        c="r",
        label=_smooth_model_label(smooth_model_poly_order, smooth_model_window),
        linewidth=2*linewidth,
    )
    yrange = get_ybounds(yrange,
                         smoothed_background_reference_spectrum[
                             np.isfinite(background_reference_spectrum)
                         ])

    axes[1].set_ylim(yrange[0] * 0.9, yrange[1] * 1.1)
    axes[1].legend(fontsize=6)
    axes[1].set_xlabel(f"Wavelength ({wave_unit})")
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
        label="Ratio of Background and Central spectra",
        linewidth=linewidth,
    )
    yrange = get_ybounds([1e50, 1e-50],
                         (background_reference_spectrum / central_reference_spectrum)[
                            np.isfinite(background_reference_spectrum)
                            * np.isfinite(central_reference_spectrum)
                        ])

    axes[2].plot(
        axis_spectral[
            np.isfinite(background_reference_spectrum)
            * np.isfinite(central_reference_spectrum)
        ],
        (
            smoothed_background_reference_spectrum
            / smoothed_reference_spectrum
        )[
            np.isfinite(background_reference_spectrum)
            * np.isfinite(central_reference_spectrum)
        ],
        c="r",
        label=_smooth_model_label(smooth_model_poly_order, smooth_model_window),
        linewidth=2*linewidth,
    )

    ratio = (
        smoothed_background_reference_spectrum / smoothed_reference_spectrum
    )[
        np.isfinite(background_reference_spectrum)
        * np.isfinite(central_reference_spectrum)
    ]
    yrange = get_ybounds(yrange, ratio)

    axes[2].set_ylim(yrange[0] * 0.9, yrange[1] * 1.1)
    axes[2].legend(fontsize=6)
    axes[2].set_xlabel(f"Wavelength ({wave_unit})")
    axes[2].set_ylabel("Background / Central\n(Flux, arbitrary)")

    fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.07, hspace=0.20)
    if isinstance(plot, PdfPages):
        plot.savefig()
    plt.close()

    return None


def build_bar_map_plot(
    plot: PdfPages | bool | None,
    frame_2d: HDUList,
    bars: np.ndarray,
    frame_2d_shortname: str,
    detector: str,
     ) -> None:
    if not plot:
        return None

    logger.info("Plotting the bars")
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    with np.errstate(invalid='ignore'):
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
    if isinstance(plot, PdfPages):
        plot.savefig()
    plt.close()

    return None


def build_spaxel_map_plot_ifu(
    plot: PdfPages | bool | None,
    axis_pixel: np.ndarray,
    axis_slice: np.ndarray,
    axis_spectral: np.ndarray,
    image: np.ndarray,
    spaxel_distance: np.ndarray,
    target_pixel: tuple[int, int],
    target_index,
    target_slices: list[int],
    frame_2d: np.ndarray,
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

    with np.errstate(invalid='ignore'):
        image_for_map = np.log10(frame_2d)

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

        xlim = (0, image_for_map.shape[1])
        
        ylim = (
            np.nanmin(trace_models[tindex][0](axis_spectral)) - 15,
            np.nanmax(trace_models[tindex][0](axis_spectral)) + 15
            )
        
        axes[tindex + 1].set_xlim(*xlim)
        axes[tindex + 1].set_ylim(*ylim)
        axes[tindex + 1].set_aspect("auto")
        axes[tindex + 1].set_title(f"Slice and Trace")

    if isinstance(plot, PdfPages):
        plot.savefig()
    plt.close()

    return None


def build_spaxel_map_plot_msa(
    plot: PdfPages | bool | None,
    axis_pixel: np.ndarray,
    axis_slice: np.ndarray,
    axis_spectral: np.ndarray,
    image: np.ndarray,
    spaxel_distance: np.ndarray,
    target_pixel: tuple[int, int],
    target_index,
    slitlet: int,
    frame_2d: HDUList,
    trace_models: list[
        list[
        Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]
        ]
    ],
    dewiggling_radius: float,
    central_reference_spectrum_radius: float,
    srcalias: str,
    conversion_factor: float = 1.,
) -> None:
    if not plot:
        return None

    logger.debug("Plotting the map and slice traces for the target pixel")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.0, 9.7),
        height_ratios=[1, 1.61],
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.05,
        top=0.93,
        hspace=0.2,
    )

    axes[0].set_title(f'Median image and slitlet cutout for\nSource {srcalias}')
    yaxis = conversion_factor * (axis_pixel - axis_pixel[target_pixel[1]])

    axes[0].scatter(yaxis, image[0, :], c='k')

    # axes[0].set_ylim(np.nanpercentile((image), 0.1), np.nanmax((image)))

    # for v in [dewiggling_radius, central_reference_spectrum_radius]:
    #     for ss in [-1, 1]:
    #         axes[0].axvline(ss * v, c='r', linestyle=':')
    # axes[0].axvline(0, 0, c="k",linestyle='--')

    axes[0].axvline(
        conversion_factor * (axis_pixel[target_index[1]] - axis_pixel[target_pixel[1]]),
        c="xkcd:orange",
        lw=2,
        label=f'Pixel {target_index[1]}'
    )

    axes[0].axvspan(
                -dewiggling_radius,
                dewiggling_radius,
                color='b',
                alpha=0.3, label='Dewiggling Region',
                edgecolor=None)

    axes[0].axvspan(
                -central_reference_spectrum_radius,
                central_reference_spectrum_radius,
                color='r',
                alpha=0.3, label='Central Spectrum region',
                edgecolor=None)

    # axes[0].set_aspect("auto")
    # axes[0].set_xlabel("Index 1, Pixel Axis, arcsec")

    axes[0].legend(fontsize=6)
    axes[0].set_ylabel("Flux\n(arbitrary)")
    axes[0].set_xlabel("Pixel Axis\n(px; index 1; relative to central pixel)")

    image_for_map = (frame_2d.data)

    axes[1].imshow(
        image_for_map,
        cmap="binary",
        origin="lower",
        vmin=np.nanpercentile(image_for_map, 10.0),
        vmax=np.nanpercentile(image_for_map, 98),
        interpolation="nearest",
        label='Slitlet image',
    )

    axes[1].plot(
        trace_models[0][1](axis_spectral),
        trace_models[0][0](axis_spectral),
        c="xkcd:orange",
        linestyle="-",
        lw=1,
        label=f'Pixel {target_index[1]} trace'
    )

    axes[1].legend(fontsize=6)
    axes[1].set_ylabel("Detector x pixel")
    axes[1].set_xlabel("Detector y pixel")
    axes[1].set_aspect("auto")

    if isinstance(plot, PdfPages):
        plot.savefig()
    plt.close()

    return None


def build_population_diagnostics(
    plot: PdfPages | bool | None,
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
) -> None:

    if not plot:
        return None

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

        _build_image_map_ifu(
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

        if isinstance(plot, PdfPages):
            plot.savefig()
        plt.close()
    return None


def get_2d_trace_model(wcs_2d: WCS,
                       im_2d: IFUImageModel,
                       n_target_traces: int = 10000) -> Callable[
                           [float, float],
                           tuple[
                               Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]],
                               Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]],
                           ],
                       ]:

    # Okay, so now we want to build a large number of forward model points

    ds = im_2d.data.shape

    target_indices = [np.random.randint(0, s, size=n_target_traces * 10) for s in ds]
    mask_2d = ~np.isnan(im_2d.data[tuple(target_indices[::-1])])

    target_indices = [s[mask_2d] for s in target_indices]

    # Now we want to do the forward pass for these indices to identify many valid RA, Dec
    world_vals = wcs_2d.pixel_to_world_values(*target_indices)

    world_lams = np.asarray(world_vals[2])
    finite_lam_mask = np.isfinite(world_lams)
    if not np.any(finite_lam_mask):
        raise ValueError("WCS returned no finite wavelength values in the forward WCS pass.")
    
    finite_lams = world_lams[finite_lam_mask]
    min_lam = np.min(finite_lams)
    max_lam = np.max(finite_lams)

    lams = np.linspace(min_lam, max_lam, 10)

    valid_sky_mask = (np.isfinite(world_vals[0]) & np.isfinite(world_vals[1]) & finite_lam_mask)
    valid_ra  = world_vals[0][valid_sky_mask][:n_target_traces]
    valid_dec = world_vals[1][valid_sky_mask][:n_target_traces]
    n_traces = len(valid_ra)

    logger.info(f"Identified {n_traces} valid traces for 2D trace modeling.")

    ra_batch  = np.repeat(valid_ra, len(lams))
    dec_batch = np.repeat(valid_dec, len(lams))
    lam_batch = np.tile(lams, n_traces)

    xy_flat = wcs_2d.world_to_pixel_values(ra_batch, dec_batch, lam_batch)
    trace_library = np.asarray(xy_flat).T.reshape(n_traces, len(lams), 2)

    V = np.vander(lams, 7)
    pf_x_lib, _, _, _ = np.linalg.lstsq(V, trace_library[:, :, 0].T, rcond=None)
    pf_y_lib, _, _, _ = np.linalg.lstsq(V, trace_library[:, :, 1].T, rcond=None)
    pf_x_lib = pf_x_lib.T
    pf_y_lib = pf_y_lib.T

    def trace_factory(ra, dec):
        d2 = (valid_ra - ra) ** 2 + (valid_dec - dec) ** 2
        closest_index = np.argmin(d2)

        pf_x = pf_x_lib[closest_index]
        pf_y = pf_y_lib[closest_index]

        def trace_x(lam_array):
            return np.polyval(pf_x, lam_array)

        def trace_y(lam_array):
            return np.polyval(pf_y, lam_array)

        return trace_y, trace_x

    return trace_factory


def _make_zero_xsol(fit_params: dict[str, int] | None) -> np.typing.NDArray[np.float64]:
    if fit_params is None:
        fit_params = {"alpha_order": 0, "beta_order": 3, "gamma_order": 0, "order": 3}
    alpha_order = fit_params["alpha_order"]
    beta_order = fit_params["beta_order"]
    gamma_order = fit_params["gamma_order"]
    order = fit_params["order"]
    return np.asarray(
        [1.0] + [0.0] * alpha_order
        + [0.0] + [0.0] * gamma_order
        + [1.0] + [0.0] * beta_order
        + [0.0, 0.0, 1.0]
        + [0.0] * (2 * order),
        dtype=float,
    )


def _plot_single_spaxel(
    x, xindex, diagnostic_fraction,
    _spaxel_world_coords, slices_to_use,
    trace_model_factory, spaxel_distance,
    axis_pixel, axis_slice, axis_spectral,
    image, target_pixel, _det_image, bars,
    dewiggling_radius, central_reference_spectrum_radius,
    _sci_data, _dq_data, _wgt_data,
    weight_based_masking_threshold, masked_regions,
    pf, pf_back, plot,
    reusable_artists,
):
    target_index = x[2]
    dist = spaxel_distance[target_index[1], target_index[0]]
    preferred_slice = x[7]
    preferred_slice_index = slices_to_use.index(preferred_slice)

    i_orig = xindex * diagnostic_fraction
    trace_models = [trace_model_factory(
        _spaxel_world_coords[preferred_slice_index][0][i_orig],
        _spaxel_world_coords[preferred_slice_index][1][i_orig],
    )]

    build_spaxel_map_plot_ifu(
        plot, axis_pixel, axis_slice, axis_spectral, image,
        spaxel_distance, target_pixel, target_index, [0],
        _det_image, bars, trace_models,
        dewiggling_radius, central_reference_spectrum_radius,
    )

    model_trace = [
        [trace_models[0][0], trace_models[0][0]],
        [trace_models[0][0], trace_models[0][0]],
    ]

    if weight_based_masking_threshold is not None:
        weight_spectrum = _wgt_data[:, target_index[0], target_index[1]]
        median_weight = np.nanmedian(weight_spectrum[weight_spectrum > 0.0])
        weight_based_mask = weight_spectrum > weight_based_masking_threshold * median_weight
    else:
        weight_based_mask = np.ones_like(axis_spectral)

    explicit_mask = np.ones_like(axis_spectral)
    for m in masked_regions:
        explicit_mask[(axis_spectral >= m[0]) & (axis_spectral <= m[1])] = 0
    weight_based_mask = (weight_based_mask * explicit_mask).astype(bool)

    xsol = x[0]
    model_details = x[1]
    reduced_x2 = x[5]
    if xsol is None:
        xsol = _make_zero_xsol(model_details)
        plot_label = f'({target_index[0]},{target_index[1]}), r = {dist:.3f}" [NO VALID FIT]'
    else:
        plot_label = f'({target_index[0]},{target_index[1]}), r = {dist:.3f}"'

    reusable_artists = plot_dewiggle_result(
        model_trace, pf, axis_spectral,
        _sci_data[:, target_index[0], target_index[1]],
        _dq_data[:, target_index[0], target_index[1]],
        saved_xsol=xsol, saved_fit_params=model_details, saved_reduced_x2=reduced_x2,
        plot=plot,
        plot_label=plot_label,
        background_pf=pf_back,
        target_maskarray=weight_based_mask,
        reusable_artists=reusable_artists,
    )
    return reusable_artists


def _plot_chunk_worker(
    chunk_idx,
    xsols_chunk,
    base_index,
    tmpdir,
    diagnostic_fraction,
    _spaxel_world_coords,
    slices_to_use,
    trace_model_factory,
    spaxel_distance,
    axis_pixel, axis_slice, axis_spectral,
    image, target_pixel, _det_image, bars,
    dewiggling_radius, central_reference_spectrum_radius,
    _sci_data, _dq_data, _wgt_data,
    weight_based_masking_threshold, masked_regions,
    pf, pf_back,
):
    import io
    import matplotlib
    matplotlib.use("Agg")

    _log_buffer = io.StringIO()
    _log_handler = logging.StreamHandler(_log_buffer)
    _log_handler.setFormatter(logging.Formatter(
        "%(asctime)s: %(funcName)-24s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    ))
    _log_handler.setLevel(logging.INFO)
    _ds_logger = logging.getLogger("hyphen.dewiggler_stack")
    _dl_logger = logging.getLogger("hyphen.dewiggle_lambda_space")
    _ds_handlers = list(_ds_logger.handlers)
    _dl_handlers = list(_dl_logger.handlers)
    _ds_propagate = _ds_logger.propagate
    _dl_propagate = _dl_logger.propagate
    for h in _ds_handlers:
        _ds_logger.removeHandler(h)
    for h in _dl_handlers:
        _dl_logger.removeHandler(h)
    _ds_logger.propagate = False
    _dl_logger.propagate = False
    _ds_logger.addHandler(_log_handler)
    _dl_logger.addHandler(_log_handler)
    _ds_logger.setLevel(logging.INFO)
    _dl_logger.setLevel(logging.INFO)

    try:
        chunk_pdf = os.path.join(tmpdir, f"chunk_{chunk_idx}.pdf")
        local_plot = PdfPages(chunk_pdf)

        reusable_artists = None
        for local_idx, x in enumerate(xsols_chunk):
            global_xindex = base_index + local_idx

            reusable_artists = _plot_single_spaxel(
                x=x, xindex=global_xindex,
                diagnostic_fraction=diagnostic_fraction,
                _spaxel_world_coords=_spaxel_world_coords,
                slices_to_use=slices_to_use,
                trace_model_factory=trace_model_factory,
                spaxel_distance=spaxel_distance,
                axis_pixel=axis_pixel, axis_slice=axis_slice,
                axis_spectral=axis_spectral, image=image,
                target_pixel=target_pixel, _det_image=_det_image, bars=bars,
                dewiggling_radius=dewiggling_radius,
                central_reference_spectrum_radius=central_reference_spectrum_radius,
                _sci_data=_sci_data, _dq_data=_dq_data, _wgt_data=_wgt_data,
                weight_based_masking_threshold=weight_based_masking_threshold,
                masked_regions=masked_regions,
                pf=pf, pf_back=pf_back,
                plot=local_plot,
                reusable_artists=reusable_artists,
            )

            if (local_idx + 1) % 100 == 0:
                logger.info(f"Chunk {chunk_idx}: rendered {local_idx + 1}/{len(xsols_chunk)}")

        local_plot.close()
    finally:
        _ds_logger.removeHandler(_log_handler)
        _dl_logger.removeHandler(_log_handler)
        _log_handler.close()
        for h in _ds_handlers:
            _ds_logger.addHandler(h)
        for h in _dl_handlers:
            _dl_logger.addHandler(h)
        _ds_logger.propagate = _ds_propagate
        _dl_logger.propagate = _dl_propagate

    return chunk_pdf, _log_buffer.getvalue()


def dewiggler_stack_ifu(
    frame_2d_filename: str,
    frame_3d_filename: str,
    output_frame_filename: str,
    target_pixel: tuple[int, int] = (-1, -1),
    plot: str | PdfPages | bool = False,
    dewiggling_radius: float = 0.5,
    central_reference_spectrum_radius: float = 0.2,
    smooth_model_poly_order: int = 5,
    smooth_model_window: int = 51,
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
    slices_to_use: list = [0],
    reference_spectra_percentile_cut: float = 1,
    try_different_slice_threshold: float = 0.025,
    weight_based_masking_threshold: Optional[float | None] = 0.5,
    fft_order_width=0.1,
    power_ratio_threshold=3.0,
    masked_regions: list = [],
    null_fit_improvement_threshold=0.005,
    px_per_slice: int = 2,
) -> object:
    """
    :type slices_to_use: list
    :param slices_to_use: This is a list of slice numbers to use for the dewiggling. We
        will estimate the slice number from the pixel location, then incrementally try
        the slices in this list. A zero here corrsponds to the estimated slice number
        and a negative number corresponds to one index lower than that estimate.
    :param frame_2d_filename: This is the path to the 2D detector-frame fits file.
    :param frame_3d_filename: This is the path to the 3D rectified fits file.
    :param output_frame_filename: This is the path to the output 3D dewiggled fits file.
    :param target_pixel: This is the central pixel associated with the point-source. If
        not provided, the code will use either the provided RA and Dec or attempt to
        find the peak of the median image.
    :param plot: This is either a filename where a resultant pdf file will be placed or
        a pdfpages object where the plots will be appended.
    :param dewiggling_radius: This is the radius in arcseconds within which the
        dewiggling will be performed.
    :param central_reference_spectrum_radius: The point-source spectrum will be
        estimated from an average of pixels within this radius.
    :param smooth_model_poly_order: The order of the polynomial used to smooth the
        reference spectra.
    :param diagnostic_fraction: For faster diagnostics, only dewiggle every Nth pixel.
        For example, if diagnostic_fraction=10, only dewiggle every 10th pixel.
    :param background_reference_spectrum_radii: The background spectrum will be
        estimated from an average of pixels between these radii.
    :param fractional_residual_threshold: If the fractional residual is above this
    threshold, the fit is considered bad.
    :param fit_param_fractional_residual_improvement_threshold: When iterating through
        the fit parameters, if the fractional residual improvement is below this
        threshold, the fit is not considered to be improved.
    :param n_jobs: The number of jobs to use for parallel processing. If plotting is
        enabled, this is ignored.
    :param perform_dewiggling: Flag whether to actually perform the dewiggling or to
        just diagnostics.
    :param target_radec: The target Ra and Dec in degrees (ICRS) corresponding to the
        point source.
    :param use_cached_wcs: Use the cached WCS if available.
    :param min_fit_params: A dictionary containing the minimum fit orders to use. The
        dictionary should include the keys 'alpha_order', 'beta_order', 'gamma_order',
        and 'order'.
    :param max_fit_params: A dictionary containing the maximum fit orders to use. The
        dictionary should include the keys 'alpha_order', 'beta_order', 'gamma_order',
        and 'order'.
    :param target_wiggle_pixels: A list of specific pixels to dewiggle (e.g., as a
        diagnostic). If not provided, the code will use all pixels within the
        dewiggling radius.
    :param mask_bad_fits: If True, replace the spectra in the output cube with NaNs if
        the fit is bad.
    :param reference_spectra_percentile_cut: When constructing the reference spectra,
        ignore pixels in the top Nth or bottom Nth percentile. 0 here corresponds to no
        cut.
    :param weight_based_masking_threshold: Perform a masking of the spectra based on
        the weights. If the weight is below this fraction of the median weight, the
        pixel will be masked.
    :param px_per_slice: The number of pixels along the slice axis per slice.
        E.g., when using a 0.05'' plate scale and the native slice size is 0.1''
        px_per_slice = 2.
    :return: The plot object containing all diagnostics.
    """

    logger.info("Initiating dewiggler...")

    frame_2d, frame_3d, frame_2d_shortname, frame_3d_shortname = load_frames(
        frame_2d_filename, frame_3d_filename
    )

    detector = determine_detector_name(frame_2d)

    if use_cached_wcs:
        logger.warning(
            "use_cached_wcs is deprecated and has no effect. "
            "The IFU WCS is now loaded directly from IFUImageModel."
        )

    frame_2d_wcs, frame_3d_wcs, _ifu_model, _cube_model = get_wcs_ifu(
        frame_2d_filename, frame_3d_filename
    )

    trace_model_factory = get_2d_trace_model(frame_2d_wcs, _ifu_model)

    axis_pixel, axis_slice, axis_spectral = get_principal_axes_ifu(frame_3d)

    spatial_arr = determine_spatial_arr(axis_pixel, axis_slice)

    plot = initialize_plot(plot)

    _general_figures_tmp = None
    if get_plot_status(plot):
        _general_figures_tmp_file = tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        )
        _general_figures_tmp_file.close()
        _general_figures_tmp = _general_figures_tmp_file.name
        _general_plot = PdfPages(_general_figures_tmp)
    else:
        _general_plot = plot

    logger.info("Constructing a median image")

    image = determine_image(frame_3d)

    target_pixel = determine_target_pixel_ifu(
        target_pixel=target_pixel,
        target_radec=target_radec,
        frame_3d_wcs=frame_3d_wcs,
        image=image,
    )

    # TODO need to change all future references to target_pixel to be [y+1, x+1]
    # instead of [x, y]
    target_pixel = (target_pixel[1] - 1, target_pixel[0] - 1)
    target_wiggle_pixels = [[t - 1 for t in tp][::-1] for tp in target_wiggle_pixels]

    spaxel_distance = determine_spaxel_distances(spatial_arr, target_pixel, conversion_factor = 3600)
    
    build_image_map_ifu(
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
            smooth_model_window=smooth_model_window,
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
            smooth_model_window=smooth_model_window,
            reference_spectra_percentile_cut=reference_spectra_percentile_cut,
        )
    )

    build_reference_spectra_plot(
        _general_plot,
        axis_spectral,
        central_reference_spectrum,
        smoothed_reference_spectrum,
        background_reference_spectrum,
        smoothed_background_reference_spectrum,
        frame_3d_shortname,
        central_reference_spectrum_radius,
        background_reference_spectrum_radii,
        smooth_model_poly_order,
        smooth_model_window=smooth_model_window,
    )

    logger.debug(f"Finding bar pixels in {frame_2d_shortname}")

    # Now we need to make a map of the bars in the 2D frame
    bars = find_bars(frame_2d[7].data)

    logger.debug(f"Found {len(bars)} bars")
    if len(bars) != 30:
        logger.error(f"Found {len(bars)} bars. This number should be 30...")

    build_bar_map_plot(_general_plot, frame_2d, bars, frame_2d_shortname, detector)

    _sci_data = np.array(frame_3d[1].data)
    _dq_data  = np.array(frame_3d[3].data)
    _wgt_data = np.array(frame_3d[4].data)
    _det_image = np.array(frame_2d[1].data)

    # Now start iterating through the wiggly pixels
    wiggly_pixels = np.where(
        (~np.isnan(image)) * (spaxel_distance.T < dewiggling_radius)
    )

    distances = spaxel_distance.T[wiggly_pixels].flatten()
    wiggly_pixels = np.asarray(wiggly_pixels)
    wiggly_pixels = wiggly_pixels[:,np.argsort(distances)]
    del distances

    if len(target_wiggle_pixels) > 0:
        wiggly_pixels = np.asarray(target_wiggle_pixels).T

    logger.info("Beginning main dewiggling loop...")
    logger.info(
        f'Found {len(wiggly_pixels[0])} wiggly pixels within {dewiggling_radius}"'
    )
    if diagnostic_fraction != 1:
        logger.info(f"Only dewiggling every {diagnostic_fraction}th pixel")


    _spaxel_world_coords = [frame_3d_wcs.pixel_to_world_values(
        wiggly_pixels[1], wiggly_pixels[0] + px_per_slice * stu, np.full(len(wiggly_pixels[0]), 1000)
    ) for stu in slices_to_use]

    # print(_spaxel_world_coords)
    def dewiggle_main_loop(i: int):

        # Ensure warning filters are active in child processes (loky/cloudpickle
        # does not re-execute module-level code when reconstructing closures)
        warnings.filterwarnings("error", message="Mean of empty slice")
        warnings.filterwarnings("error", message="invalid value encountered in scalar divide")

        # Capture logs from this worker invocation
        import io
        _log_buffer = io.StringIO()
        _log_handler = logging.StreamHandler(_log_buffer)
        _log_handler.setFormatter(logging.Formatter(
            "%(asctime)s: %(funcName)-24s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %I:%M:%S %p",
        ))
        _log_handler.setLevel(logging.INFO)
        _ds_logger = logging.getLogger("hyphen.dewiggler_stack")
        _dl_logger = logging.getLogger("hyphen.dewiggle_lambda_space")
        # Save and remove existing handlers to prevent double logging in workers
        _ds_handlers = list(_ds_logger.handlers)
        _dl_handlers = list(_dl_logger.handlers)
        _ds_propagate = _ds_logger.propagate
        _dl_propagate = _dl_logger.propagate
        for h in _ds_handlers:
            _ds_logger.removeHandler(h)
        for h in _dl_handlers:
            _dl_logger.removeHandler(h)
        _ds_logger.propagate = False
        _dl_logger.propagate = False
        _ds_logger.addHandler(_log_handler)
        _dl_logger.addHandler(_log_handler)
        _ds_logger.setLevel(logging.INFO)
        _dl_logger.setLevel(logging.INFO)

        try:
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

            # target_slices = [0]

            logger.debug(
                f"This spaxel is estimated to be in slice {base_target_slice + 1} "
                f"(fractional position {fp[0]:.2f})"
            )

            logger.debug(
                f"For this spaxel, we will try fitting with slices"
                f"{[base_target_slice + 1 + t for t in slices_to_use]}"
            )

            if weight_based_masking_threshold is not None:
                weight_spectrum = _wgt_data[:, target_index[0], target_index[1]]
                median_weight = np.nanmedian(weight_spectrum[weight_spectrum > 0.0])
                weight_based_mask = (
                    weight_spectrum > weight_based_masking_threshold * median_weight
                )

            else:
                logger.debug("Weight-based masking is disabled.")
                weight_based_mask = np.ones_like(axis_spectral)

            explicit_mask = np.ones_like(axis_spectral)
            if len(masked_regions) > 0:
                logger.debug("Applying masked regions")
                for m in masked_regions:
                    logger.info("Applying mask {m[0]}-{m[1]}")
                    explicit_mask[
                        (axis_spectral >= m[0]) & (axis_spectral <= m[1])
                    ] = 0

            weight_based_mask = weight_based_mask * explicit_mask
            weight_based_mask = weight_based_mask.astype(bool)

            # find world coordinates for the target pixel
            trace_models = list([trace_model_factory(_spaxel_world_coords[ii][0][i], _spaxel_world_coords[ii][1][i]) for ii in range(len(_spaxel_world_coords))])

            build_spaxel_map_plot_ifu(
                False,
                axis_pixel,
                axis_slice,
                axis_spectral,
                image,
                spaxel_distance,
                target_pixel,
                target_index,
                [0],
                _det_image,
                bars,
                trace_models,
                dewiggling_radius,
                central_reference_spectrum_radius,
            )

            # For good measure, we'll try the de-wiggling on the slice before and after the
            # target slice and use the best model of the three.

            base_x2 = 1e10
            preferred_wiggle_model = np.ones_like(axis_spectral) * np.nan
            preferred_dewiggled_spectrum = np.ones_like(axis_spectral) * np.nan
            preferred_wiggle_fft = {}
            preferred_target_slice = 0
            preferred_xsol = None
            preferred_model_details = min_fit_params.copy()
            preferred_null_x2 = float('inf')


            logger.debug(f"Starting the dewiggler for index {i}!")
            for tindex, target_slice in enumerate(slices_to_use):
                if base_x2**0.5 < try_different_slice_threshold:
                    continue

                logger.info(
                    40 * "-~"
                )
                logger.info(
                    f"  Current √𝜈 = {base_x2**0.5:.4f} > "
                    f"{try_different_slice_threshold:.4f}"
                )
                logger.info(f"  Trying trace from slice {target_slice}")
                logger.info(
                    40 * "-~"
                )

                model_trace = [
                    [trace_models[tindex][0], trace_models[tindex][0]],
                    [trace_models[tindex][0], trace_models[tindex][0]],
                ]

                if null_fit_improvement_threshold >= 0:
                    logger.info("  Fitting null model with f = sqrt(2)")
                    null_wiggle_model, null_dewiggled_spectrum, null_x2, null_xsol, null_model_details, null_wiggle_fft = (
                        dewiggle_lambda_space(
                            model_trace,
                            pf,
                            axis_spectral,
                            _sci_data[:, target_index[0], target_index[1]],
                            _dq_data[:, target_index[0], target_index[1]],
                            plot=False,
                            plot_label=(
                                f'({target_index[0]},{target_index[1]}), r = {dist:.3f}", '
                                f'Slice {target_slice}'
                            ),
                            background_pf=pf_back,
                            min_fit_params=min_fit_params,
                            max_fit_params=max_fit_params,
                            sqrt_nu_improvement_threshold=(
                                fit_param_fractional_residual_improvement_threshold
                            ),
                            target_maskarray=weight_based_mask,
                            frequency_fudge=2**0.5,
                        )
                    )
                    logger.info(f" Found √𝜈 = {null_x2**0.5:.4f} with null model (f = sqrt(2))")
                else:
                    null_x2 = float('inf')
                    logger.info("  Null model test disabled (null_fit_improvement_threshold < 0)")

                logger.info("  Now fitting with f = 1")
                wiggle_model, dewiggled_spectrum, x2, xsol, model_details, wiggle_fft = (
                    dewiggle_lambda_space(
                        model_trace,
                        pf,
                        axis_spectral,
                        _sci_data[:, target_index[0], target_index[1]],
                        _dq_data[:, target_index[0], target_index[1]],
                        plot=False,
                        plot_label=(
                            f'({target_index[0]},{target_index[1]}), r = {dist:.3f}", '
                            f'Slice {target_slice}'
                        ),
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
                logger.info(f"  Found √𝜈 = {x2**0.5:.4f} using slice {target_slice}")
                logger.info(f"  Previous best was {base_x2**0.5:.4f}")
                logger.info(f"Current fit orders are (O, 𝛼, 𝛽, 𝛾) = ({model_details['order']}, "
                    f"{model_details['alpha_order']}, {model_details['beta_order']}, "
                    f"{model_details['gamma_order']})"
                )
                if model_details['order'] == 0:
                    logger.info("  No wiggle terms were used in this fit.")
                logger.info(
                    40 * "-~"
                )

                if null_fit_improvement_threshold >= 0:
                    improved_over_null = ((null_x2**0.5 - x2**0.5) > null_fit_improvement_threshold)
                    logger.info(f"  Fit with f=1 is {'better' if improved_over_null else 'not better'} than null model by more than {null_fit_improvement_threshold:.4f} in √𝜈 improvement")
                    logger.info(f"  {'Accepting' if improved_over_null else 'Rejecting'} this fit based on null model comparison")
                else:
                    improved_over_null = True
                    logger.info("  Null model test disabled — accepting fit unconditionally")

                if ((x2 < base_x2) or (preferred_wiggle_model is None)) and improved_over_null:
                    base_x2 = x2
                    preferred_wiggle_model = wiggle_model
                    preferred_dewiggled_spectrum = dewiggled_spectrum
                    preferred_wiggle_fft = wiggle_fft
                    preferred_target_slice = target_slice
                    preferred_xsol = xsol
                    preferred_model_details = model_details
                    preferred_null_x2 = null_x2

                else:
                    if preferred_xsol is None:
                        preferred_dewiggled_spectrum = _sci_data[:, target_index[0], target_index[1]]

            # ── Vitals summary for the adopted fit ──────────────────────────
            logger.info("")
            logger.info(60 * "=")
            logger.info(
                f"  ADOPTED FIT SUMMARY  spaxel ({target_index[0]}, {target_index[1]})"
            )
            logger.info(60 * "=")
            logger.info(f"  Adopted trace index : {preferred_target_slice}")
            if preferred_xsol is None:
                logger.info("  No valid fit was obtained for this spaxel.")
            else:
                _md = preferred_model_details
                _ao = _md["alpha_order"]
                _bo = _md["beta_order"]
                _go = _md["gamma_order"]
                _wo = _md["order"]
                _cP_idx  = _ao + _bo + _go + 3
                _cG_idx  = _ao + _bo + _go + 4
                _f_idx   = _ao + _bo + _go + 5
                _ab_start = _ao + _bo + _go + 6
                _xsol = preferred_xsol

                logger.info(f"  Fit orders          : O={_wo}, α={_ao}, β={_bo}, γ={_go}")

                # Wiggle terms
                if _wo == 0:
                    logger.info("  Wiggle terms        : (none — order 0)")
                else:
                    logger.info("  Wiggle terms:")
                    for _j in range(_wo):
                        _a_j = _xsol[_ab_start + 2 * _j]
                        _b_j = _xsol[_ab_start + 2 * _j + 1]
                        _amp = (_a_j**2 + _b_j**2)**0.5
                        logger.info(
                            f"    order {_j+1:2d} :  a={_a_j:+.6f}  b={_b_j:+.6f}"
                            f"  |amplitude|={_amp:.6f}"
                        )

                # Beta (wiggle-envelope) polynomial
                _beta_start = _ao + _go + 2
                _beta_coeffs = _xsol[_beta_start : _beta_start + _bo + 1]
                logger.info(
                    f"  Wiggle envelope β   : [{', '.join(f'{c:+.6f}' for c in _beta_coeffs)}]"
                )

                # Alpha (point-source continuum) polynomial
                _alpha_coeffs = _xsol[0 : _ao + 1]
                logger.info(
                    f"  α continuum         : [{', '.join(f'{c:+.6f}' for c in _alpha_coeffs)}]"
                )

                # Gamma (background continuum) polynomial
                _gamma_coeffs = _xsol[_ao + 1 : _ao + _go + 2]
                logger.info(
                    f"  γ background        : [{', '.join(f'{c:+.6f}' for c in _gamma_coeffs)}]"
                )

                # Scalar params
                logger.info(f"  cP                  : {_xsol[_cP_idx]:+.6f}")
                logger.info(f"  cG                  : {_xsol[_cG_idx]:+.6f}")
                logger.info(f"  f                   : {_xsol[_f_idx]:.6f}")

            # Diagnostics
            logger.info(f"  √𝜈 (adopted fit)    : {base_x2**0.5:.6f}")
            if null_fit_improvement_threshold >= 0:
                logger.info(
                    f"  √𝜈 (null test)      : {preferred_null_x2**0.5:.6f}"
                )
            else:
                logger.info("  √𝜈 (null test)      : (test disabled)")
            logger.info(60 * "=")
            logger.info("")
            # ── End vitals summary ───────────────────────────────────────────

            return (
                preferred_xsol,
                preferred_model_details,
                target_index,
                preferred_dewiggled_spectrum,
                preferred_wiggle_fft,
                base_x2,
                _log_buffer.getvalue(),
                preferred_target_slice,
            )
        finally:
            _ds_logger.removeHandler(_log_handler)
            _dl_logger.removeHandler(_log_handler)
            _log_handler.close()
            # Restore original handlers and propagate settings
            for h in _ds_handlers:
                _ds_logger.addHandler(h)
            for h in _dl_handlers:
                _dl_logger.addHandler(h)
            _ds_logger.propagate = _ds_propagate
            _dl_logger.propagate = _dl_propagate

    xsol_fn = output_frame_filename.replace(".fits", ".pkl")

    if perform_dewiggling:

        total_steps = len(range(len(wiggly_pixels[0]))[::diagnostic_fraction])
        xsols = ProgressParallel(
            n_jobs=n_jobs, verbose=0, prefer="processes", total=total_steps
        )(
            delayed(dewiggle_main_loop)(i)
            for i in range(len(wiggly_pixels[0]))[::diagnostic_fraction]
        )
        if get_plot_status(plot):
            _save_fit_results(xsols, xsol_fn)
    else:
        xsols = _load_fit_results(xsol_fn)

    _flush_spaxel_logs(xsols)

    # Plot pass: generate per-spaxel diagnostic plots from saved results
    # (runs after parallel fitting completes, or in plot-only mode)
    if get_plot_status(plot):
        logger.info("Plotting: regenerating diagnostic plots from saved fit results")

        n_chunks = max(1, min(n_jobs, len(xsols)))
        chunk_size = (len(xsols) + n_chunks - 1) // n_chunks

        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_results = ProgressParallel(
                n_jobs=n_jobs, verbose=0, prefer="processes", total=n_chunks
            )(
                delayed(_plot_chunk_worker)(
                    chunk_idx=ci,
                    xsols_chunk=xsols[ci * chunk_size : (ci + 1) * chunk_size],
                    base_index=ci * chunk_size,
                    tmpdir=tmpdir,
                    diagnostic_fraction=diagnostic_fraction,
                    _spaxel_world_coords=_spaxel_world_coords,
                    slices_to_use=slices_to_use,
                    trace_model_factory=trace_model_factory,
                    spaxel_distance=spaxel_distance,
                    axis_pixel=axis_pixel, axis_slice=axis_slice,
                    axis_spectral=axis_spectral, image=image,
                    target_pixel=target_pixel, _det_image=_det_image, bars=bars,
                    dewiggling_radius=dewiggling_radius,
                    central_reference_spectrum_radius=central_reference_spectrum_radius,
                    _sci_data=_sci_data, _dq_data=_dq_data, _wgt_data=_wgt_data,
                    weight_based_masking_threshold=weight_based_masking_threshold,
                    masked_regions=masked_regions,
                    pf=pf, pf_back=pf_back,
                )
                for ci in range(n_chunks)
            )

            chunk_pdfs = [r[0] for r in chunk_results]
            chunk_logs = [r[1] for r in chunk_results]

            chunk_pdfs.sort(
                key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])
            )

            final_path = plot._file.fh.name

            _general_plot.close()
            if isinstance(plot, PdfPages):
                plot.close()

            from pypdf import PdfWriter

            merged_tmp = os.path.join(tmpdir, "merged_final.pdf")
            merger = PdfWriter()
            if _general_figures_tmp is not None:
                merger.append(_general_figures_tmp)
            for chunk_path in chunk_pdfs:
                merger.append(chunk_path)
            merger.write(merged_tmp)
            merger.close()
            if _general_figures_tmp is not None:
                os.unlink(_general_figures_tmp)
            shutil.move(merged_tmp, final_path)

            logger.info(f"Merged {len(chunk_pdfs)} chunk PDFs into {final_path}")

            root_handlers = [
                h for h in logging.getLogger().handlers
                if isinstance(h, logging.FileHandler)
            ]
            if root_handlers:
                stream = root_handlers[0].stream
                for log_text in chunk_logs:
                    if log_text:
                        stream.write(log_text)
                        if not log_text.endswith("\n"):
                            stream.write("\n")
                stream.flush()

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

            # First, make sure that 'freq' is in the dictionary:
            if 'freq' in x[4].keys():
                fft_freq = x[4]['freq']
                p1_indices = (
                    (fft_freq[:] >= 1 - fft_order_width) &
                    (fft_freq[:] <= 1 + fft_order_width)
                )

                if sum(p1_indices) == 0:
                    p1_indices[np.argmin(np.abs(fft_freq - 1.0))] = True

                p1 = np.nanmean(np.abs(x[4]['residual_fft'][p1_indices]))
                pavg = np.nanmean(np.abs(x[4]['residual_fft']))
                logger.info('')

                if p1 / pavg > power_ratio_threshold:
                    bad_fit = True

                logger.info(
                    f"Pixel ({target_index[0]}, {target_index[1]}) has √𝜈 = "
                    f"{fit_x2**0.5:.4f}."
                    f"    and P(1) / P(3/2) = {p1 / pavg:.4f}."
                )
            else:
                bad_fit = True

            if mask_bad_fits & bad_fit:
                logger.warning("Masking this pixel.")

                if retry_allowed:
                    try:
                        frame_3d[1].data[:, target_index[0], target_index[1]] = \
                            preferred_dewiggled_spectrum
                        x = dewiggle_main_loop(xindex)
                    except Exception as e:
                        logger.warning(
                            f"Error in dewiggle_main_loop: {e}"
                        )
                        preferred_dewiggled_spectrum = (
                            np.ones_like(preferred_dewiggled_spectrum) * np.nan
                        )

                logger.info(
                    f"Pixel ({target_index[0]}, {target_index[1]}) has √𝜈 = "
                    f"{fit_x2**0.5:.4f}."
                    f"    and P(1) / P(3/2) = {p1 / pavg:.4f}."
                )
            else:
                logger.info(
                    f"Pixel ({target_index[0]}, {target_index[1]}) has √𝜈 = "
                    f"{fit_x2**0.5:.4f}."
                    f"    and P(1) / P(3/2) = {p1 / pavg:.4f}."
                )

            logger.info('')

            frame_3d[1].data[:, target_index[0], target_index[1]] = \
                preferred_dewiggled_spectrum

    logger.info("Saving dewiggled frame")

    frame_3d.writeto(output_frame_filename, overwrite=True)

    logger.info("Dewiggling complete!")

    close_plot(plot)

    return plot


def identify_slitlets(frame_2d, frame_3d, srcalias):
    for slitlet_2d in range(len(frame_2d)):
        # line_src = frame_2d[slitlet_2d].header.get('SRCALIAS', 'UNKNOWN')
        # print(f'x{line_src}x')
        if frame_2d[slitlet_2d].header.get('SRCALIAS', 'UNKNOWN') == str(srcalias):
            break
    
    for slitlet_3d in range(len(frame_3d)):
        if frame_3d[slitlet_3d].header.get('SRCALIAS', 'UNKNOWN') == str(srcalias):
            break
    
    return int((slitlet_2d - 1) / 10), int((slitlet_3d - 1) / 8)


def dewiggler_stack_msa(
    frame_2d_filename: str,
    frame_3d_filename: str,
    output_frame_filename: str,
    target_pixel: int = -1,
    plot: str | PdfPages | bool = False,
    dewiggling_radius: float = 0.5,
    slitlet: int = 0,
    central_reference_spectrum_radius: float = 0.2,
    smooth_model_poly_order: int = 5,
    diagnostic_fraction: int = 1,
    background_reference_spectrum_radii: tuple[float, float] = (1.0, 1.5),
    fractional_residual_threshold: float = 0.05,
    fit_param_fractional_residual_improvement_threshold: float = 0.001,
    n_jobs: int = 1,
    perform_dewiggling: bool = True,
    target_radec: tuple[float, float] = (0.0, 0.0),
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
    slices_to_use: list = [0],
    reference_spectra_percentile_cut: float = 1,
    try_different_slice_threshold: float = 0.025,
    weight_based_masking_threshold: Optional[float | None] = 0.5,
    fft_order_width=0.1,
    power_ratio_threshold=3.0,
    masked_regions: list = [],
    srcalias: str = 'UNKNOWN',
) -> object:
    """
    :type slices_to_use: list
    :param slices_to_use: This is a list of slice numbers to use for the dewiggling. We
        will estimate the slice number from the pixel location, then incrementally try
        the slices in this list. A zero here corrsponds to the estimated slice number
        and a negative number corresponds to one index lower than that estimate.
    :param frame_2d_filename: This is the path to the 2D detector-frame fits file.
    :param frame_3d_filename: This is the path to the 3D rectified fits file.
    :param output_frame_filename: This is the path to the output 3D dewiggled fits file.
    :param target_pixel: This is the central pixel associated with the point-source. If
        not provided, the code will use either the provided RA and Dec or attempt to
        find the peak of the median image.
    :param plot: This is either a filename where a resultant pdf file will be placed or
        a pdfpages object where the plots will be appended.
    :param dewiggling_radius: This is the radius in arcseconds within which the
        dewiggling will be performed.
    :param central_reference_spectrum_radius: The point-source spectrum will be
        estimated from an average of pixels within this radius.
    :param smooth_model_poly_order: The order of the polynomial used to smooth the
        reference spectra.
    :param diagnostic_fraction: For faster diagnostics, only dewiggle every Nth pixel.
        For example, if diagnostic_fraction=10, only dewiggle every 10th pixel.
    :param background_reference_spectrum_radii: The background spectrum will be
        estimated from an average of pixels between these radii.
    :param fractional_residual_threshold: If the fractional residual is above this
    threshold, the fit is considered bad.
    :param fit_param_fractional_residual_improvement_threshold: When iterating through
        the fit parameters, if the fractional residual improvement is below this
        threshold, the fit is not considered to be improved.
    :param n_jobs: The number of jobs to use for parallel processing. If plotting is
        enabled, this is ignored.
    :param perform_dewiggling: Flag whether to actually perform the dewiggling or to
        just diagnostics.
    :param target_radec: The target Ra and Dec in degrees (ICRS) corresponding to the
        point source.
    :param use_cached_wcs: Use the cached WCS if available.
    :param min_fit_params: A dictionary containing the minimum fit orders to use. The
        dictionary should include the keys 'alpha_order', 'beta_order', 'gamma_order',
        and 'order'.
    :param max_fit_params: A dictionary containing the maximum fit orders to use. The
        dictionary should include the keys 'alpha_order', 'beta_order', 'gamma_order',
        and 'order'.
    :param target_wiggle_pixels: A list of specific pixels to dewiggle (e.g., as a
        diagnostic). If not provided, the code will use all pixels within the
        dewiggling radius.
    :param mask_bad_fits: If True, replace the spectra in the output cube with NaNs if
        the fit is bad.
    :param reference_spectra_percentile_cut: When constructing the reference spectra,
        ignore pixels in the top Nth or bottom Nth percentile. 0 here corresponds to no
        cut.
    :param weight_based_masking_threshold: Perform a masking of the spectra based on
        the weights. If the weight is below this fraction of the median weight, the
        pixel will be masked.
    :return: The plot object containing all diagnostics.
    """

    logger.info("Initiating dewiggler...")

    frame_2d, frame_3d, frame_2d_shortname, frame_3d_shortname = load_frames(
        frame_2d_filename, frame_3d_filename
    )

    slitlet_2d, slitlet_3d = identify_slitlets(frame_2d, frame_3d, srcalias)

    frame_2d_wcs, frame_3d_wcs, frame_2d_model, frame_3d_model = get_wcs_msa(
        frame_2d_filename, frame_3d_filename)

    axis_pixel, axis_slice, axis_spectral = get_principal_axes_msa(
                                                frame_3d, frame_3d_wcs, slitlet_3d)

    spatial_arr = determine_spatial_arr(axis_pixel, axis_slice)

    plot = initialize_plot(plot)

    logger.info("Constructing a median image")

    image = determine_image(frame_3d, frame_index=8 * slitlet_3d + 1, collapse_axis=1)

    target_pixel = determine_target_pixel_msa(
        target_pixel=target_pixel,
        target_radec=target_radec,
        frame_3d_wcs=frame_3d_wcs,
        image=image,
    )

    # TODO need to change all future references to target_pixel to be [y+1, x+1]
    # instead of [x, y]

    target_pixel = (target_pixel[1] - 1, target_pixel[0] - 1)
    target_wiggle_pixels = [[t - 1 for t in tp][::-1] for tp in target_wiggle_pixels]

    spaxel_distance = determine_spaxel_distances(spatial_arr,
                                                 target_pixel,
                                                 conversion_factor=1.0)

    # interpret the background radii.

    if -1 in background_reference_spectrum_radii:
        background_reference_spectrum_radii = (-1, -1)

    build_image_map_msa(
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
            data_frame_index=8 * slitlet_3d + 1,
            data_quality_frame_index=None,
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
            data_frame_index=8 * slitlet_3d + 1,
            data_quality_frame_index=None,
        )
    )

    _pf_back_failed = (
        not isinstance(pf_back, InterpolatedPf)
        and np.any(np.isnan(pf_back))
    )
    if _pf_back_failed:
        pf_back = np.zeros_like(pf_back)
        smoothed_background_reference_spectrum = np.zeros_like(
            smoothed_background_reference_spectrum
        )
        background_reference_spectrum = np.zeros_like(
            background_reference_spectrum
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
        smooth_model_window=smooth_model_window,
        unit='px',
    )

    # Now start iterating through the wiggly pixels
    wiggly_pixels = np.where(
        (~np.isnan(image)) * (spaxel_distance.T <= dewiggling_radius)
    )

    if len(target_wiggle_pixels) > 0:
        wiggly_pixels = np.asarray(target_wiggle_pixels).T

    logger.info("Beginning main dewiggling loop...")
    logger.info(
        f'Found {len(wiggly_pixels[0])} wiggly pixels within {dewiggling_radius} px'
    )
    if diagnostic_fraction != 1:
        logger.info(f"Only dewiggling every {diagnostic_fraction}th pixel")

    def dewiggle_main_loop(i: int):

        # Ensure warning filters are active in child processes (loky/cloudpickle
        # does not re-execute module-level code when reconstructing closures)
        warnings.filterwarnings("error", message="Mean of empty slice")
        warnings.filterwarnings("error", message="invalid value encountered in scalar divide")

        # Capture logs from this worker invocation
        import io
        _log_buffer = io.StringIO()
        _log_handler = logging.StreamHandler(_log_buffer)
        _log_handler.setFormatter(logging.Formatter(
            "%(asctime)s: %(funcName)-24s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %I:%M:%S %p",
        ))
        _log_handler.setLevel(logging.INFO)
        _ds_logger = logging.getLogger("hyphen.dewiggler_stack")
        _dl_logger = logging.getLogger("hyphen.dewiggle_lambda_space")
        # Save and remove existing handlers to prevent double logging in workers
        _ds_handlers = list(_ds_logger.handlers)
        _dl_handlers = list(_dl_logger.handlers)
        _ds_propagate = _ds_logger.propagate
        _dl_propagate = _dl_logger.propagate
        for h in _ds_handlers:
            _ds_logger.removeHandler(h)
        for h in _dl_handlers:
            _dl_logger.removeHandler(h)
        _ds_logger.propagate = False
        _dl_logger.propagate = False
        _ds_logger.addHandler(_log_handler)
        _dl_logger.addHandler(_log_handler)
        _ds_logger.setLevel(logging.INFO)
        _dl_logger.setLevel(logging.INFO)

        try:
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

            logger.debug("Weight-based masking is disabled.")
            weight_based_mask = np.ones_like(axis_spectral)
            
            explicit_mask = np.ones_like(axis_spectral)
            if len(masked_regions) > 0:
                logger.debug("Applying masked regions")
                for m in masked_regions:
                    logger.info(f"Applying mask {m[0]}-{m[1]}")
                    explicit_mask[
                        (axis_spectral >= m[0]) & (axis_spectral <= m[1])
                    ] = 0

            weight_based_mask = weight_based_mask * explicit_mask
            weight_based_mask = weight_based_mask.astype(bool)

            world_coords = frame_3d_wcs[slitlet_3d].pixel_to_world(
                int(len(axis_spectral) / 2.), target_index[1]
            )

            trace_models = [
                get_trace_model(
                    frame_2d_wcs[slitlet_2d],
                    world_coords[0].ra.deg,
                    world_coords[0].dec.deg
                )
            ]

            build_spaxel_map_plot_msa(
                False,
                axis_pixel,
                axis_slice,
                axis_spectral,
                image,
                spaxel_distance,
                target_pixel,
                target_index,
                None,
                frame_2d[10 * slitlet_2d + 1],
                trace_models,
                dewiggling_radius,
                central_reference_spectrum_radius,
                srcalias,
            )

            base_x2 = 1e10
            preferred_wiggle_model = np.ones_like(axis_spectral) * np.nan
            preferred_dewiggled_spectrum = np.ones_like(axis_spectral) * np.nan
            preferred_wiggle_fft = {}

            logger.debug(f"Starting the dewiggler for index {i}!")
            # for tindex, target_slice in enumerate(target_slices):
            # if base_x2**0.5 < try_different_slice_threshold:
            #     continue

            logger.info(
                40 * "-~"
            )
            # logger.info(
            #     f"  Current √𝜈 = {base_x2**0.5:.4f} > "
            #     f"{try_different_slice_threshold:.4f}"
            # )
            logger.info(f"  Trying trace for srcalias {srcalias}")
            logger.info(
                40 * "-~"
            )

            model_trace = [
                [trace_models[0][0], trace_models[0][0]],
                [trace_models[0][0], trace_models[0][0]],
            ]

            wiggle_model, dewiggled_spectrum, x2, xsol, model_details, wiggle_fft = (
                dewiggle_lambda_space(
                    model_trace,
                    pf,
                    axis_spectral,
                    frame_3d[8 * slitlet_3d + 1].data[target_index[1], :],
                    None,
                    plot=False,
                    plot_label=(
                        f'Pixel {target_index[1]}; R = {dist:.1f} px; '
                        f'Src {srcalias}'
                    ),
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
            logger.info(f"  Found √𝜈 = {x2**0.5:.4f} using slice {slitlet_3d + 1}")
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
                _log_buffer.getvalue(),
            )
        finally:
            _ds_logger.removeHandler(_log_handler)
            _dl_logger.removeHandler(_log_handler)
            _log_handler.close()
            # Restore original handlers and propagate settings
            for h in _ds_handlers:
                _ds_logger.addHandler(h)
            for h in _dl_handlers:
                _dl_logger.addHandler(h)
            _ds_logger.propagate = _ds_propagate
            _dl_logger.propagate = _dl_propagate

    xsol_fn = output_frame_filename.replace(".fits", ".pkl")

    if perform_dewiggling:

        total_steps = len(range(len(wiggly_pixels[0]))[::diagnostic_fraction])
        xsols = ProgressParallel(
            n_jobs=n_jobs, verbose=0, prefer="processes", total=total_steps
        )(
            delayed(dewiggle_main_loop)(i)
            for i in range(len(wiggly_pixels[0]))[::diagnostic_fraction]
        )
        if get_plot_status(plot):
            _save_fit_results(xsols, xsol_fn)
    else:
        xsols = _load_fit_results(xsol_fn)

    _flush_spaxel_logs(xsols)

    # Plot pass: generate per-spaxel diagnostic plots from saved results
    # (runs after parallel fitting completes, or in plot-only mode)
    if get_plot_status(plot):
        logger.info("Plotting: regenerating diagnostic plots from saved fit results")
        reusable_artists = None

        for xindex in range(len(xsols)):
            x = xsols[xindex]
            target_index = x[2]
            dist = spaxel_distance[target_index[1], target_index[0]]

            world_coords = frame_3d_wcs[slitlet_3d].pixel_to_world(
                int(len(axis_spectral) / 2.), target_index[1]
            )
            trace_models = [
                get_trace_model(
                    frame_2d_wcs[slitlet_2d],
                    world_coords[0].ra.deg,
                    world_coords[0].dec.deg,
                )
            ]

            build_spaxel_map_plot_msa(
                plot, axis_pixel, axis_slice, axis_spectral, image,
                spaxel_distance, target_pixel, target_index,
                None,
                frame_2d[10 * slitlet_2d + 1],
                trace_models,
                dewiggling_radius, central_reference_spectrum_radius,
                srcalias,
            )

            model_trace = [
                [trace_models[0][0], trace_models[0][0]],
                [trace_models[0][0], trace_models[0][0]],
            ]

            weight_based_mask = np.ones_like(axis_spectral)
            explicit_mask = np.ones_like(axis_spectral)
            if len(masked_regions) > 0:
                for m in masked_regions:
                    explicit_mask[
                        (axis_spectral >= m[0]) & (axis_spectral <= m[1])
                    ] = 0
            weight_based_mask = weight_based_mask * explicit_mask
            weight_based_mask = weight_based_mask.astype(bool)

            reusable_artists = plot_dewiggle_result(
                model_trace,
                pf,
                axis_spectral,
                frame_3d[8 * slitlet_3d + 1].data[target_index[1], :],
                None,
                saved_xsol=x[0],
                saved_fit_params=x[1],
                saved_reduced_x2=x[5],
                plot=plot,
                plot_label=f'Pixel {target_index[1]}; R = {dist:.1f} px; Src {srcalias}',
                background_pf=pf_back,
                target_maskarray=weight_based_mask,
                reusable_artists=reusable_artists,
            )

            if (xindex + 1) % 100 == 0:
                logger.info(f"Plot-only: rendered {xindex + 1}/{len(xsols)} spaxels")

        logger.info(f"Plot-only: rendered all {len(xsols)} spaxels")

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

            # First, make sure that 'freq' is in the dictionary:
            if 'freq' in x[4].keys():
                fft_freq = x[4]['freq']
                p1_indices = (
                    (fft_freq[:] >= 1 - fft_order_width) &
                    (fft_freq[:] <= 1 + fft_order_width)
                )

                if sum(p1_indices) == 0:
                    p1_indices[np.argmin(np.abs(fft_freq - 1.0))] = True

                p1 = np.nanmean(np.abs(x[4]['residual_fft'][p1_indices]))
                pavg = np.nanmean(np.abs(x[4]['residual_fft']))
                logger.info('')

                if p1 / pavg > power_ratio_threshold:
                    bad_fit = True

                logger.info(
                    f"Pixel ({target_index[0]}, {target_index[1]}) has √𝜈 = "
                    f"{fit_x2**0.5:.4f}."
                    f"    and P(1) / P(3/2) = {p1 / pavg:.4f}."
                )
            else:
                bad_fit = True

            if mask_bad_fits & bad_fit:
                logger.warning("Masking this pixel.")

                if retry_allowed:
                    try:
                        frame_3d[8 * slitlet_3d + 1].data[target_index[1], :] = \
                            preferred_dewiggled_spectrum
                        x = dewiggle_main_loop(xindex)
                    except Exception as e:
                        logger.warning(
                            f"Error in dewiggle_main_loop: {e}"
                        )
                        preferred_dewiggled_spectrum = (
                            np.ones_like(preferred_dewiggled_spectrum) * np.nan
                        )

                else:
                    preferred_dewiggled_spectrum = (
                        np.ones_like(preferred_dewiggled_spectrum) * np.nan
                    )

        # frame_3d[1].data[
        #     :, target_index[0], target_index[1]
        # ] = preferred_dewiggled_spectrum

        frame_3d[8 * slitlet_3d + 1].data[target_index[1], :] = preferred_dewiggled_spectrum
    logger.info("Dewiggling complete!")

    # build_population_diagnostics(
    #     plot,
    #     axis_pixel,
    #     axis_slice,
    #     image,
    #     spaxel_distance,
    #     target_pixel,
    #     frame_3d_shortname,
    #     dewiggling_radius,
    #     central_reference_spectrum_radius,
    #     background_reference_spectrum_radii,
    #     xsols,
    #     wiggly_pixels,
    #     diagnostic_fraction=diagnostic_fraction,
    # )
    close_plot(plot)

    frame_3d.writeto(output_frame_filename, overwrite=True)

    return plot


def parse_yaml_arguments_and_dewiggle_ifu(getter):

    input_2d_prototype = getter("input_2d_filename_prototype")
    input_3d_prototype = getter("input_3d_filename_prototype")
    output_filename_prototype = getter("output_filename_prototype")
    target_detectors = getter("target_detectors")
    target_frames = getter("target_frames")

    if not target_frames or target_frames == [""]:
        raise ValueError(
            "No target frames specified and I don't yet know how to handle that..."
        )

    log_filename_prototype = (
        output_filename_prototype.replace(".gz", "").replace(".fits", "") + ".log"
    )
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

            dewiggler_stack_ifu(
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
                smooth_model_window=getter("smooth_model_window"),
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
                masked_regions=getter("masked_regions"),
                mask_bad_fits=bool(getter("mask_bad_fits")),
                slices_to_use=getter("slices_to_use"),
                reference_spectra_percentile_cut=getter(
                    "reference_spectra_percentile_cut"
                ),
                try_different_slice_threshold=getter("try_different_slice_threshold"),
                null_fit_improvement_threshold=getter("null_fit_improvement_threshold"),
            )

    return 0


def parse_yaml_arguments_and_dewiggle_msa(getter):

    input_2d_prototype = getter("input_2d_filename_prototype")
    input_3d_prototype = getter("input_3d_filename_prototype")
    output_filename_prototype = getter("output_filename_prototype")
    target_detectors = getter("target_detectors")
    target_frames = getter("target_frames")

    assert target_frames is not [
        ""
    ], "No target frames specified and I don't yet know how to handle that..."

    log_filename_prototype = (
        output_filename_prototype.replace(".gz", "").replace(".fits", "") + ".log"
    )
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

            dewiggler_stack_msa(
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
                smooth_model_window=getter("smooth_model_window"),
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
                min_fit_params=getter("min_fit_params"),
                max_fit_params=getter("max_fit_params"),
                target_wiggle_pixels=getter("target_wiggly_pixels"),
                masked_regions=getter("masked_regions"),
                mask_bad_fits=bool(getter("mask_bad_fits")),
                slices_to_use=getter("slices_to_use"),
                reference_spectra_percentile_cut=getter(
                    "reference_spectra_percentile_cut"
                ),
                try_different_slice_threshold=getter("try_different_slice_threshold"),
                srcalias=getter("srcalias"),
            )

    return 0


def parse_yaml_arguments_and_batch_dewiggle_ifu(getter):

    # This function should loop through the list contained in 'central_pixel' and for
    # each element build a modified getter plugging in appropriate modifications to
    # the output filenames and central pixel and pass that getter to the single dewiggle
    # parser.

    # This function should transparently pass most all arguments from getter along to
    # the various single-dewiggle passes.
    # The exceptions are
    # 1. central_pixel must be provided here and must be a list of 2-tuples
    # 2. input_3d_filename_prototype will be modified to chain the output of one
    #    step to the next
    # 3. output_filename_prototype will be modified similar to inptu_3d_filename
    # 4. plot_filename_prototype similar.

    input_3d_prototype = getter("input_3d_filename_prototype")

    central_pixels = getter("central_pixel")

    output_filename_prototypes = [
        getter("output_filename_prototype").replace('PIXEL', f'{c[0]}_{c[1]}')
        for c in central_pixels
    ]
    plot_filename_prototype = getter("plot_filename_prototype")

    if plot_filename_prototype:
        plot_filename_prototypes = [
            plot_filename_prototype.replace("PIXEL", f"{c[0]}_{c[1]}")
            for c in central_pixels
        ]
    else:
        plot_filename_prototypes = [False for c in central_pixels]

    input_3d_prototypes = (
        [getter("input_3d_filename_prototype")] + output_filename_prototypes[:-1]
    )

    for (input_3d_prototype, output_filename_prototype,
            plot_filename_prototype, central_pixel) in zip(
            input_3d_prototypes, output_filename_prototypes, plot_filename_prototypes,
            central_pixels):

        override = {
            'input_3d_filename_prototype': input_3d_prototype,
            'output_filename_prototype': output_filename_prototype,
            'plot_filename_prototype': plot_filename_prototype,
            'central_pixel': central_pixel,
        }

        partial_getter = partial(getter, override=[override])

        parse_yaml_arguments_and_dewiggle_ifu(partial_getter)

    return 0

def parse_yaml_arguments_and_batch_dewiggle_msa(getter):

    # This function should loop through the list contained in 'srcalias' and for
    # each element build a modified getter plugging in appropriate modifications to
    # the output filenames and central pixel and pass that getter to the single dewiggle
    # parser.

    # This function should transparently pass most all arguments from getter along to
    # the various single-dewiggle passes.
    # The exceptions are
    # 1. srcalias must be provided here and must be a list of strs
    # 2. input_3d_filename_prototype will be modified to chain the output of one
    #    step to the next
    # 3. output_filename_prototype will be modified similar to input_3d_filename
    # 4. plot_filename_prototype similar.

    srcalias = getter("srcalias")

    output_filename_prototypes = [
        getter("output_filename_prototype").replace('SRC', f'{src}')
        for src in srcalias
    ]
    plot_filename_prototype = getter("plot_filename_prototype")

    if plot_filename_prototype:
        plot_filename_prototypes = [
            plot_filename_prototype.replace("SRC", f"{src}")
            for src in srcalias
        ]
    else:
        plot_filename_prototypes = [False for src in srcalias]

    input_3d_prototypes = (
        [getter("input_3d_filename_prototype")] + output_filename_prototypes[:-1]
    )

    for (input_3d_prototype, output_filename_prototype,
            plot_filename_prototype, src) in zip(
            input_3d_prototypes, output_filename_prototypes, plot_filename_prototypes,
            srcalias):

        override = {
            'input_3d_filename_prototype': input_3d_prototype,
            'output_filename_prototype': output_filename_prototype,
            'plot_filename_prototype': plot_filename_prototype,
            'srcalias': src,
        }

        partial_getter = partial(getter, override=[override])

        parse_yaml_arguments_and_dewiggle_msa(partial_getter)

    return 0


single_dewiggle_ifu_step = {
    "target_step_type": "dewiggle_ifu",
    "target_function": parse_yaml_arguments_and_dewiggle_ifu,
    "default_options": single_dewiggle_ifu_default_options,
    "necessary_arguments": single_dewiggle_ifu_necessary_arguments,
}

batch_dewiggle_ifu_step = {
    "target_step_type": "batch_dewiggle_ifu",
    "target_function": parse_yaml_arguments_and_batch_dewiggle_ifu,
    "default_options": batch_dewiggle_ifu_default_options,
    "necessary_arguments": batch_dewiggle_ifu_necessary_arguments,
}

single_dewiggle_msa_step = {
    "target_step_type": "dewiggle_msa",
    "target_function": parse_yaml_arguments_and_dewiggle_msa,
    "default_options": single_dewiggle_msa_default_options,
    "necessary_arguments": single_dewiggle_msa_necessary_arguments,
}

batch_dewiggle_msa_step = {
    "target_step_type": "batch_dewiggle_msa",
    "target_function": parse_yaml_arguments_and_batch_dewiggle_msa,
    "default_options": batch_dewiggle_msa_default_options,
    "necessary_arguments": batch_dewiggle_msa_necessary_arguments,
}


def dewiggler_wrapper(yaml_file):

    ingest_yaml(yaml_file, [single_dewiggle_ifu_step, batch_dewiggle_ifu_step,
                            single_dewiggle_msa_step, batch_dewiggle_msa_step])

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
