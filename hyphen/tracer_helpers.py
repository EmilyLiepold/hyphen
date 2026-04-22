import numpy as np
from scipy.interpolate import interp1d



def find_bars(image, min_pixels=1000):
    from scipy.ndimage import label

    ## This function takes an image which has np.nan values in regions between bars and returns a list of pixels belonging to each bar.
    ## The image is assumed to be 2D, with the second dimension being roughly aligned with the wavelength axis and the first axis crossing the bars.
    ## Uses 8-connected component labeling so that isolated masked (NaN) pixels within a bar do not split it into multiple components.

    mask = ~np.isnan(image)

    # 8-connectivity: diagonally touching pixels belong to the same component
    structure = np.ones((3, 3), dtype=int)
    labeled, n_bars = label(mask, structure=structure)

    full_bars = []
    for bar_id in range(1, n_bars + 1):
        rows, cols = np.where(labeled == bar_id)
        if len(rows) >= min_pixels:
            full_bars.append(np.column_stack([rows, cols]).astype(int))

    # Sort bars by mean row position (cross-dispersion axis)
    full_bars.sort(key=lambda b: b[:, 0].mean())

    return full_bars

def trace_pixels(bar, waveimage, target_pixel=0.5):

    wave = waveimage[bar[:, 0], bar[:, 1]]
    ax1_vals = sorted(list(set(bar[:, 1])))

    trace = []
    for ax1_val in ax1_vals:
        target_indices = np.where(bar[:, 1] == ax1_val)[0]

        target_ax0 = bar[target_indices, 0]
        target_wave = wave[target_indices]

        trace_ax0 = np.min(target_ax0) + target_pixel * (
            np.max(target_ax0) - np.min(target_ax0)
        )

        trace_wave = interp1d(target_ax0, target_wave, bounds_error=False)(trace_ax0)
        trace.append([trace_ax0, ax1_val, trace_wave])

    trace = np.asarray(trace)
    return trace


def trace_model(trace, order=3):

    pf_ax1 = np.polyfit(trace[:, 2], trace[:, 1], order)
    pf_ax0 = np.polyfit(trace[:, 2], trace[:, 0], order)

    pv_ax1 = np.polyval(pf_ax1, np.linspace(min(trace[:, 1]), max(trace[:, 1]), 100000))
    pv_ax0 = np.polyval(pf_ax0, np.linspace(min(trace[:, 1]), max(trace[:, 1]), 100000))

    inverse_ax1 = interp1d(
        pv_ax1, np.linspace(min(trace[:, 1]), max(trace[:, 1]), 100000)
    )
    inverse_ax0 = interp1d(
        pv_ax0, np.linspace(min(trace[:, 1]), max(trace[:, 1]), 100000)
    )

    return ((lambda x: np.polyval(pf_ax0, x)), inverse_ax0), (
        (lambda x: np.polyval(pf_ax1, x)),
        inverse_ax1,
    )


def infer_fractional_position(image, target_position):
    ##
    ## First we want to find the largest and smallest indices along both axes which are not nan
    slices = [image[:, target_position[1]], image[target_position[0], :]]
    slice_indices = [
        np.where(~np.isnan(slices[0]))[0],
        np.where(~np.isnan(slices[1]))[0],
    ]
    slice_extrema = [
        [min(slice_indices[0]), max(slice_indices[0])],
        [min(slice_indices[1]), max(slice_indices[1])],
    ]

    fractional_position = [
        (target_position[0] - slice_extrema[0][0])
        / (slice_extrema[0][1] - slice_extrema[0][0]),
        (target_position[1] - slice_extrema[1][0])
        / (slice_extrema[1][1] - slice_extrema[1][0]),
    ]
    return fractional_position

    pass
