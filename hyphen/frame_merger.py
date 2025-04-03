import argparse
import logging
from typing import Optional

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .yaml_helpers import ingest_yaml
from .helpers import wigglebar

logger = logging.getLogger(__name__)


def identify_chip_gap(
    weight: np.typing.NDArray[np.float64],
    weight_based_masking_threshold: Optional[float] = 0.5,
) -> tuple[int, int]:
    """
    Identify the longest contiguous gap of zero or NaN values in the weight array.

    :param weight: A numpy array of float64 values representing weights.
    :return:  A tuple containing the start and end indices of the longest contiguous
        gap of zero or NaN values.
    """

    gaps = []
    in_gap = False

    median_weight = np.nanmedian(weight)

    good_weight = ~(
        np.isnan(weight) |
        (weight == 0) |
        (weight < median_weight * weight_based_masking_threshold)
    )

    current_gap = []
    for i, gw in enumerate(good_weight):
        if not gw and not in_gap:
            current_gap = [i]
            in_gap = True
        elif not gw and in_gap:
            current_gap.append(i)
        elif gw and in_gap:
            gaps.append(current_gap[:])
            current_gap = []
            in_gap = False
        if i == len(weight) - 1 and in_gap:
            gaps.append(current_gap)

    gap_lengths = np.asarray([len(g) for g in gaps])
    long_gap = gaps[np.argmax(gap_lengths)]

    return long_gap[0], long_gap[-1]


def load_frames(
    primary_fns: list[list[str]],
    base_fn: str = "",
    weights_fns: Optional[list[list[str]]] = None,
    load_frames_from_disk: bool = False,
    ) -> tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        list[list[fits.HDUList]]
        ]:

    """
    Load frames from FITS files and optionally cache the results to disk.

    :param primary_fns: list[list[str]]
        A list of lists containing the file paths to the primary FITS files.
    :param base_fn: str
        The base filename for caching the arrays to disk. Default is an empty string.
    :param weights_fns: list[list[str]] or None, optional
        A list of lists containing the file paths to the weights FITS files. If None,
        the weights will be loaded from the primary files. Default is None.
    :param load_frames_from_disk: bool, optional
        If True, load the frames from disk even if cached arrays are available. Default
        is False.

    :return:
        A tuple containing the following elements:
        - fluxes (np.typing.NDArray[np.float64]): The flux arrays.
        - variances (np.typing.NDArray[np.float64]): The variance arrays.
        - dqs (np.typing.NDArray[np.float64]): The data quality arrays.
        - weights (np.typing.NDArray[np.float64]): The weight arrays.
        - frames (list[list[fits.HDUList]]): The list of FITS HDUList objects.
    """

    # Read in the data
    frames = [[fits.open(fn) for fn in ff] for ff in primary_fns]
    if weights_fns is not None:
        frames_for_weights = [[fits.open(fn) for fn in ff] for ff in weights_fns]
    else:
        frames_for_weights = None

    fluxes = np.empty(1, dtype=np.float64)
    variances = np.empty(1, dtype=np.float64)
    dqs = np.empty(1, dtype=np.float64)
    weights = np.empty(1, dtype=np.float64)

    try:
        logger.info("Trying to load arrays from cached arrays")
        arrs = np.load(f"{base_fn}_cached.npz")
        fluxes = arrs["fluxes"]
        variances = arrs["variances"]
        dqs = arrs["dqs"]
        weights = arrs["weights"]
        del arrs

    except FileNotFoundError:
        logger.warning("Unable to load arrays from cache. Loading from disk instead")
        load_frames_from_disk = True

    if load_frames_from_disk:
        logger.info("Loading arrays from disk")
        try:

            logger.info("   Loading dq arrays")
            dqs = np.asarray([[frame[3].data for frame in fr] for fr in frames])

            logger.info("   Loading flux arrays")
            fluxes = np.asarray([[frame[1].data for frame in fr] for fr in frames])
            fluxes[dqs != 0] = np.nan
            fluxes = np.nansum(fluxes, axis=1)

            logger.info("   Loading var arrays")
            variances = np.asarray([[frame[2].data for frame in fr] for fr in frames])
            variances[dqs != 0] = np.nan
            variances = np.nansum(variances, axis=1)

            logger.info("   Loading weight arrays")
            if frames_for_weights is not None:
                weights = np.asarray(
                    [[frame[4].data for frame in fr] for fr in frames_for_weights]
                )
            else:
                weights = np.asarray([[frame[4].data for frame in fr] for fr in frames])
            weights[dqs != 0] = np.nan
            weights = np.nansum(weights, axis=1)
            dqs = np.nanmin(dqs, axis=1)

            logger.info("   Caching arrays to disk")
            np.savez_compressed(f"{base_fn}_cached.npz",
                                fluxes=fluxes,
                                variances=variances,
                                dqs=dqs,
                                weights=weights)

        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Error {e} in file loading")

    return fluxes, variances, dqs, weights, frames


def determine_gap_edges(
    weights, 
    base_fn="", 
    recompute_gaps=False, 
    weight_based_masking_threshold=0.5,
):

    gap_edges = None

    try:
        logger.info("Trying to load gap array from cache")
        gap_edges = np.load(f"{base_fn}_gaps.npy").T

    except FileNotFoundError:
        recompute_gaps = True

    if recompute_gaps:
        try:
            gap_edges = np.asarray(
                [
                    [
                        [
                            identify_chip_gap(
                                weights[i, :, j, k],
                                weight_based_masking_threshold=weight_based_masking_threshold,
                            )
                            for i in range(weights.shape[0])
                        ]
                        for j in range(weights.shape[2])
                    ]
                    for k in range(weights.shape[3])
                ]
            )
            np.save(f"{base_fn}_gaps.npy", gap_edges)
            gap_edges = gap_edges.T
        except Exception as e:
            assert False, f"Error {e} in gap calculation"

    return gap_edges


def build_merge_per_spaxel_diagnostics(
    pdf, raw_fluxes, fluxes, raw_weights, weights, title=""
):
    fig, axes = plt.subplots(2, figsize=(10, 10))

    for i in range(raw_fluxes.shape[0]):
        axes[0].plot(
            raw_fluxes[i],
            label=f"{i}",
            c=f"C{i}",
            alpha=1,
        )

        axes[0].plot(fluxes, c="k")

        axes[1].plot(
            raw_weights[i],
            label=f"{i}",
            c=f"C{i}",
            alpha=1,
        )
        axes[1].plot(weights, c="k")

    axes[0].set_title(title)

    pdf.savefig()
    plt.close()

    return pdf


def build_spaxel_spectra_diagnostics(pdf, raw_fluxes, title=""):
    fig, axes = plt.subplots(1, figsize=(10, 10))

    for i in range(raw_fluxes.shape[0]):

        axes.plot(
            raw_fluxes[i] / np.nanmedian(raw_fluxes[i][raw_fluxes[i] > 0]) - i,
            label=f"{i}",
            c=f"C{i}",
            alpha=1,
        )

    axes.set_ylim(-(raw_fluxes.shape[0] + 1), 2)

    axes.legend(loc="lower right")
    axes.set_title(title)
    pdf.savefig()
    plt.close()

    return pdf


def build_big_spectra_figure(
    pdf,
    lam,
    spectra,
    r,
    indices,
    bottom_label_room=1.0,
    left_label_room=1.0,
    panel_width=6.0,
    offset=0.75,
):

    print(lam.shape)
    print(spectra.shape)
    number_to_plot = spectra.shape[0]

    # bottom_label_room = 1
    height = number_to_plot * 0.25 + bottom_label_room
    width = panel_width + left_label_room

    fig, axes = plt.subplots(1, figsize=(width, height))

    for j in wigglebar(range(number_to_plot)):

        norm_val = np.nanmedian(spectra[j])
        normed_flux = spectra[j] / norm_val
        normed_flux[normed_flux[:] == 0] = np.nan

        axes.plot(lam, normed_flux - offset * j, c="k")
        axes.text(
            lam[0],
            normed_flux[0] - offset * j,
            f"R = {r[j]:.3f}'', ({indices[j, 0]}, {indices[j, 1]})\n"
            f"norm = {norm_val:.2f}",
            verticalalignment="center",
            horizontalalignment="right",
            fontsize=6,
        )

    axes.set_ylim(-offset * number_to_plot, 2 * offset)
    axes.set_xlim(lam[0], lam[-1])
    axes.set_yticks([])
    fig.subplots_adjust(
        top=1.0,
        right=1.0,
        left=left_label_room / width,
        bottom=bottom_label_room / height,
    )
    pdf.savefig()
    plt.close()

    return pdf


def frame_merge(
    fnss,
    out_fn,
    pixel_cut_distance=0,
    diagnostic_plot_filename=None,
    center_pixel_for_plots=None,
    target_radius=0.5,
    load_frames_from_disk=False,
    weights_fns=None,
    nrs1_region=(250, 1500),
    nrs2_region=(2500, 3000),
    mask_gap_union=True,
    recompute_gaps=False,
    gap_region=(1750, 2150),
    mask_file=None,
    target_detectors=["nrs1", "nrs2"],
    weight_based_masking_threshold=0.5,
):
    if diagnostic_plot_filename == "":
        pdf = None
        logger.info(
            "No diagnostic plot filename provided. Not saving diagnostic plots."
        )
    else:
        pdf = PdfPages(diagnostic_plot_filename)
        logger.info(f"Saving diagnostic plots to {diagnostic_plot_filename}")

    # We want to follow the same protocol as the
    # jwst.cube_build.ifu_cube.find_spaxel_flux() function.

    base_fn = out_fn.replace('.gz','').replace('.fits','')

    fluxes, variances, dqs, weights, frames = load_frames(
        fnss,
        base_fn=base_fn,
        weights_fns=weights_fns,
        load_frames_from_disk=load_frames_from_disk,
    )

    if pixel_cut_distance != 0:
        logger.info(f"Masking {pixel_cut_distance} around each dq != 0 voxel.")

        for pindex in range(pixel_cut_distance):
            dqs[:, 1:, :, :] += dqs[:, :-1, :, :]
            dqs[:, :-1, :, :] += dqs[:, 1:, :, :]

            dqs[:, :, 1:, :] += dqs[:, :, :-1, :]
            dqs[:, :, :-1, :] += dqs[:, :, 1:, :]

            dqs[:, :, :, 1:] += dqs[:, :, :, :-1]
            dqs[:, :, :, :-1] += dqs[:, :, :, 1:]

    # Identify voxels which are 'good'
    # These pixels have
    # 1. dq == 0
    # 2. Positive weight
    # 3. Finite flux
    # 4. Numerical flux
    # 5. Nonzero flux

    good_dq = (
        (dqs[...] == 0)
        & (weights[...] > 0)
        & np.isfinite(fluxes[...])
        & ~np.isnan(fluxes[...])
        & (fluxes[...] != 0)
    )

    # If a voxel is bad, then mark its flux and weight with nan.
    weights[~good_dq] = np.nan
    fluxes[~good_dq] = np.nan
    del good_dq

    if mask_gap_union:
        logger.info("Masking the union of the chip gap region across all frames.")

        # Now we'd like to find the pixels which bound the chip gap for each spaxel
        # of each frame.
        gap_edges = determine_gap_edges(
            weights, 
            base_fn=base_fn, 
            recompute_gaps=recompute_gaps,
            weight_based_masking_threshold=weight_based_masking_threshold,
        )

        # Cast to float so that we can assign NaN
        gap_edges = gap_edges.astype(float)

        # Make sure that the gap regions we've identified align with the gap.
        # Do this by ignoring any which starts before the spot where we thinkn
        # the gap should appear or ends after that spot.

        if 'nrs1' not in target_detectors:
            gap_region[0] = -1000000
        if 'nrs2' not in target_detectors:
            gap_region[1] = 1000000

        for i in range(weights.shape[0]):
            for j in range(weights.shape[2]):
                for k in range(weights.shape[3]):

                    if gap_edges[0, i, j, k] < gap_region[0]:
                        weights[i, :, j, k] = np.nan
                        gap_edges[0, i, j, k] = np.nan
                    elif gap_edges[0, i, j, k] > gap_region[1]:
                        weights[i, :, j, k] = np.nan
                        gap_edges[0, i, j, k] = np.nan

                    if gap_edges[1, i, j, k] > gap_region[1]:
                        weights[i, :, j, k] = np.nan
                        gap_edges[1, i, j, k] = np.nan
                    elif gap_edges[1, i, j, k] < gap_region[0]:
                        weights[i, :, j, k] = np.nan
                        gap_edges[1, i, j, k] = np.nan

        # Now, we want to find the minimum and maximum valid gap starting index
        # across the N frames
        min_gap_edge = np.nanmin(gap_edges[0, ...], axis=0).astype(int)
        max_gap_edge = np.nanmax(gap_edges[1, ...], axis=0).astype(int)

        if pdf is not None:

            for j in range(gap_edges.shape[1]):
                fig, axes = plt.subplots(2, figsize=(10, 10))
                fig.suptitle(f"Gap edges for frame {j}")

                axes[0].imshow(
                    gap_edges[0, j, ...], origin="lower", vmin=1750, vmax=2500
                )
                axes[1].imshow(
                    gap_edges[1, j, ...], origin="lower", vmin=1750, vmax=2500
                )

                pdf.savefig()
                plt.close()

            fig, axes = plt.subplots(2, figsize=(10, 10))
            fig.suptitle("Extreme gap edges across all frames")
            axes[0].imshow(min_gap_edge, origin="lower", vmin=1750, vmax=2500)
            axes[1].imshow(max_gap_edge, origin="lower", vmin=1750, vmax=2500)
            pdf.savefig()
            plt.close()

        for i in range(weights.shape[2]):
            for j in range(weights.shape[3]):
                weights[:, min_gap_edge[i, j] : max_gap_edge[i, j], i, j] = np.nan
                fluxes[:, min_gap_edge[i, j] : max_gap_edge[i, j], i, j] = np.nan

    # We want to first determine the flux and variance per contributing spaxel.

    # TODO This should really be done by reading off from the files rather than
    # hardcoding a pixel size
    # TODO this region should also be a bit more documented and cleaned.
    xaxis = 0.05 * (np.arange(fluxes.shape[2]) - center_pixel_for_plots[1])
    yaxis = 0.05 * (np.arange(fluxes.shape[3]) - center_pixel_for_plots[0])
    lams = (
        np.arange(fluxes.shape[1]) * frames[0][0][1].header["CDELT3"]
        + frames[0][0][1].header["CRVAL3"]
    )

    r = np.sqrt(xaxis[:, None] ** 2 + yaxis[None, :] ** 2)
    xindex = np.arange(r.shape[0])
    yindex = np.arange(r.shape[1])
    xys = np.array(np.meshgrid(xindex, yindex)).T

    flat_r = r.reshape(-1)
    flat_x = xys[:, :, 0].reshape(-1)
    flat_y = xys[:, :, 1].reshape(-1)

    number_within_target_radius = sum(flat_r[:] <= target_radius)

    rsort = np.argsort(flat_r)

    # Additionally, we may want to mask some spectra by hand.
    # This is kind of awkwardly done by specifying indices after sorting.
    # e.g., index 0 is the center, and 1 is neighboring

    print("Doing additional Masks")
    if mask_file is not None:
        mask = open(mask_file).readlines()
        for mask_line in mask:
            mask_line_vals = [int(a) for a in mask_line.split()]
            px_index = mask_line_vals[0]
            frame_index = mask_line_vals[1]

            fluxes[frame_index, :, flat_x[rsort[px_index]], flat_y[rsort[px_index]]] = (
                np.nan
            )
            weights[
                frame_index, :, flat_x[rsort[px_index]], flat_y[rsort[px_index]]
            ] = np.nan

            logger.info(
                f"Masking {frame_index}: "
                f"({flat_x[rsort[px_index]]}, {flat_y[rsort[px_index]]})"
            )

    # Now we want to determine whether we have data from both detectors.
    # We'll do this by checking that the product of the fluxes in broad regions in the
    # middle of each detector are all nan or all 0.

    nrs1_data = ~np.logical_or(
        np.prod(fluxes[:, nrs1_region[0] : nrs1_region[1], ...] == 0, axis=1),
        np.prod(np.isnan(fluxes[:, nrs1_region[0] : nrs1_region[1], ...]), axis=1),
    )
    nrs2_data = ~np.logical_or(
        np.prod(fluxes[:, nrs2_region[0] : nrs2_region[1], ...] == 0, axis=1),
        np.prod(np.isnan(fluxes[:, nrs2_region[0] : nrs2_region[1], ...]), axis=1),
    )

    nrs1_data = np.logical_or(nrs1_data, ('nrs1' not in target_detectors))
    nrs2_data = np.logical_or(nrs2_data, ('nrs2' not in target_detectors))

    have_both_sets = np.logical_and(nrs1_data, nrs2_data)

    # When we don't have both sets of data for a spaxel, mask it out for both detectors.
    for i in range(fluxes.shape[0]):
        for j in range(fluxes.shape[2]):
            for k in range(fluxes.shape[3]):
                if not have_both_sets[i, j, k]:
                    fluxes[i, :, j, k] = np.nan
                    weights[i, :, j, k] = np.nan

    total_masked = 0

    for j in range(number_within_target_radius):
        total_masked += np.sum(~have_both_sets[:, flat_x[rsort[j]], flat_y[rsort[j]]])
        for k in range(have_both_sets.shape[0]):
            if not have_both_sets[k, flat_x[rsort[j]], flat_y[rsort[j]]]:
                logger.info(
                    f"Missing data at {k}: ({flat_x[rsort[j]]}, {flat_y[rsort[j]]})"
                )

    logger.info(f"Total masked pixels = {total_masked}")

    # Store a copy of the fluxes and weights before we start merging frames.

    raw_fluxes = fluxes.copy()
    raw_weights = weights.copy()

    fluxes = fluxes * weights
    variances = variances * weights * weights

    fluxes = np.nansum(fluxes, axis=0)
    variances = np.nansum(variances, axis=0)
    weights = np.nansum(weights, axis=0)

    # Now we want to normalize by the number of contributing pixels
    good = weights > 0
    fluxes[good] = fluxes[good] / weights[good]
    variances[good] = variances[good] / (weights[good] * weights[good])

    if pdf is not None:

        build_big_spectra_figure(
            pdf,
            lams,
            fluxes[
                :,
                flat_x[rsort[:number_within_target_radius]],
                flat_y[rsort[:number_within_target_radius]],
            ].T,
            r[
                flat_x[rsort[:number_within_target_radius]],
                flat_y[rsort[:number_within_target_radius]],
            ],
            np.asarray(
                [
                    flat_x[rsort[:number_within_target_radius]],
                    flat_y[rsort[:number_within_target_radius]],
                ]
            ).T,
        )

        for j in wigglebar(range(number_within_target_radius)):
            build_merge_per_spaxel_diagnostics(
                pdf,
                raw_fluxes[:, :, flat_x[rsort[j]], flat_y[rsort[j]]],
                fluxes[:, flat_x[rsort[j]], flat_y[rsort[j]]],
                raw_weights[:, :, flat_x[rsort[j]], flat_y[rsort[j]]],
                weights[:, flat_x[rsort[j]], flat_y[rsort[j]]],
                title=(
                    f'pixel {flat_x[rsort[j]], flat_y[rsort[j]]}, '
                    f'R = {r[flat_x[rsort[j]], flat_y[rsort[j]]]}"'
                ),
            )

            build_spaxel_spectra_diagnostics(
                pdf,
                raw_fluxes[:, :, flat_x[rsort[j]], flat_y[rsort[j]]],
                title=(
                    f'index {j}, pixel {flat_x[rsort[j]], flat_y[rsort[j]]}, '
                    f'R = {r[flat_x[rsort[j]], flat_y[rsort[j]]]}"'
                ),
            )

        pdf.close()

    # Now we want to write the data to a new file by modifying the first frame

    # Update the data
    frames[0][0][1].data = fluxes[:]
    frames[0][0][2].data = variances[:]
    frames[0][0][4].data = weights[:]

    # Update the header
    frames[0][0][0].header["NFRAMES"] = len(fnss)

    # Write the file
    frames[0][0].writeto(out_fn, overwrite=True)

    return 0


def find_targeted_file_name(input_filenames, target_frame, target_detector):
    input_base_names = [g.split("/")[-1] for g in input_filenames]

    for i, j in zip(input_base_names, input_filenames):
        if (target_frame in i) and (target_detector in i):
            return j

    return None


def parse_yaml_arguments_and_merge(getter):

    input_filename_prototype = getter("input_filename_prototype")
    target_detectors = getter("target_detectors")

    target_frames = getter("target_frames")

    assert target_frames is not [
        ""
    ], "No target frames specified and I don't yet know how to handle that..."

    frame_name_list = [
        [
            input_filename_prototype.replace("DETECTOR", target_detector).replace(
                "FRAME", target_frame
            )
            for target_detector in target_detectors
        ]
        for target_frame in target_frames
    ]

    output_filename = getter("output_filename")

    log_filename = output_filename.replace(".gz", "").replace(".fits", "") + ".log"

    logger = logging.getLogger()

    with open(log_filename, "w"):
        pass

    logging.basicConfig(
        format="%(asctime)s: %(name)s.%(funcName)s(): %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p",
        filename=log_filename,
        level=logging.INFO,
        force=True,
    )

    logger.info(f'Beginning frame merger task {getter("label", "")}')

    frame_merge(
        frame_name_list,
        output_filename,
        pixel_cut_distance=int(getter("pixel_cut_distance")),
        diagnostic_plot_filename=getter("plot_filename"),
        center_pixel_for_plots=getter("central_pixel"),
        target_radius=float(getter("radius_for_diagnostics")),
        load_frames_from_disk=bool(getter("force_reload_frames")),
        nrs1_region=getter("nrs1_region"),
        nrs2_region=getter("nrs2_region"),
        mask_gap_union=getter("mask_gap_union"),
        recompute_gaps=bool(getter("force_reload_gaps")),
        gap_region=getter("gap_region"),
        mask_file=getter("mask_filename"),
        target_detectors=target_detectors,
    )

    return 0


default_options = {
    "input_filename_prototype": None,
    "target_detectors": ["nrs1", "nrs2"],
    "target_frames": [""],
    "output_filename": None,
    "plot_filename": None,
    "pixel_cut_distance": 0,
    "central_pixel": None,
    "radius_for_diagnostics": 0.5,
    "force_reload_frames": False,
    "force_reload_gaps": False,
    "mask_filename": None,
    "nrs1_region": [250, 1500],
    "nrs2_region": [2500, 3000],
    "gap_region": [1750, 2150],
    "mask_gap_union": True,
}

necessary_arguments = (
    "label",
    "input_filename_prototype",
    "output_filename",
    "target_frames",
)

step = {
    "target_step_type": "merge",
    "necessary_arguments": necessary_arguments,
    "default_options": default_options,
    "target_function": parse_yaml_arguments_and_merge,
}


def frame_merge_wrapper(yaml_file):

    ingest_yaml(yaml_file, [step])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge frames")
    parser.add_argument(
        "yaml_file",
        type=str,
        help="The yaml file which contains the arguments for the frame merger",
    )
    args = parser.parse_args()

    frame_merge_wrapper(args.yaml_file)
