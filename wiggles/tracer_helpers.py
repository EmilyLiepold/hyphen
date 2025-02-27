import numpy as np
from scipy.interpolate import interp1d


def find_bars(image, max_mean_deviation=2.0):

    ## This function takes an image which has np.nan values in regions between bars and returns a list of pixels belonging to each bar
    ## The image is assumed to be 2D, with the second dimension being roughly aligned with the wavelength axis and the first axis crossing the bars

    # for each column in the image, associate the contiguous pixels with the same bar.
    bars = []
    # At the end of this loop, 'bars' will be a list of lists.
    # Each element of 'bars' will correspond to a column in the image
    # Each sub-element of 'bars' will contain a list of lists of pixel indices corresponding to contiguous sets of pixels in the column
    for j in range(image.shape[1]):
        # find the non-nan pixels in the column
        valid_pixels = sorted(list(np.where(~np.isnan(image[:, j]))[0]))
        bar_sets = []
        bar = []

        # associate each of the non-nan pixels with a proto-bar.
        while len(valid_pixels) > 0:
            bar.append(valid_pixels[0])
            valid_pixels.remove(bar[-1])

            # If the next pixel in our collection isn't contiguous then end the bar and start a new one.
            if (len(valid_pixels) > 0) and (valid_pixels[0] != (bar[-1] + 1)):
                bar_sets.append(bar)
                bar = []

        # If we'd started a bar but ran out of pixels, finish that bar
        if len(bar) > 0:
            bar_sets.append(bar)
        bars.append(bar_sets)

    # Now we want to associate the bars from each columnwith each other.
    # Start out with a dummy bar that's well outside the range of any real bar.
    full_bars = [[1e8, 1e8]]
    full_bar_means = [1e8]

    # Start looping through the columns
    for j in range(len(bars)):

        # We only want to associate each bar with one other bar, so we'll keep track of which bars are still available to be associated.
        associable_indices = [i for i in range(len(full_bars))]

        ## find the mean pixel locations of each bar
        bar_means = [np.mean(b) for b in bars[j]]

        ## loop through the bars in this column
        while len(bars[j]) > 0:
            bar = bars[j].pop(0)
            bar_mean = bar_means.pop(0)

            ## find the closest bar
            closest_bar = np.argmin(np.abs(bar_mean - np.asarray(full_bar_means)))

            ## if it's valid to associate the bars, do so
            if (
                np.abs(bar_mean - full_bar_means[closest_bar]) < max_mean_deviation
            ) and (closest_bar in associable_indices):

                for bb in bar:
                    full_bars[closest_bar].append([bb, j])
                full_bar_means[closest_bar] = bar_mean
                associable_indices.remove(closest_bar)

            ## Otherwise start a new bar!
            else:
                full_bars.append([])
                for bb in bar:
                    full_bars[-1].append([bb, j])
                full_bar_means.append(bar_mean)

    ## remove the dummy bar
    full_bars.pop(0)
    full_bar_means.pop(0)

    ## convert the pixel indices to integer arrays
    full_bars = [np.asarray(b).astype(int) for b in full_bars]

    ## sort the bars by their mean pixel location
    full_bar_means = [np.mean(b[:, 0]) for b in full_bars]
    full_bars = [full_bars[x] for x in np.argsort(full_bar_means)]

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
