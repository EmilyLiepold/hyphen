from tqdm import tqdm
from joblib import Parallel
from scipy.stats import sigmaclip
import numpy as np
import matplotlib.pyplot as plt


def wigglebar(*args, **kwargs):
    if not args:
        return tqdm(
            **kwargs,
            bar_format=(
                "|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
                "{rate_fmt}{postfix}]"
            ),
            ascii="~◊-",
        )
    return tqdm(
        *args,
        **kwargs,
        bar_format=(
            "|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
        ascii="~◊-",
    )


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with wigglebar(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def get_ybounds(yrange, spectrum):
    out_yrange = [y for y in yrange]
    clipped_spectrum = sigmaclip(spectrum[np.isfinite(spectrum)])

    out_yrange[0] = min(yrange[0], np.float64(clipped_spectrum[1])) - np.finfo(np.float64).eps
    out_yrange[1] = max(yrange[1], np.float64(clipped_spectrum[2])) + np.finfo(np.float64).eps

    return out_yrange
