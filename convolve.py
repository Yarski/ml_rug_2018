""" Functions adapted from:
    https://bic-berkeley.github.io/psych-214-fall-2016/convolution_background.html
    Use these to convolve a Hemodynamic Response Function (HRF) to
    movie event data. """

import numpy as np
from scipy.stats import gamma


def hrf(times):
    """ Return values for HRF at given times """

    peak_values = gamma.pdf(times, 6)
    undershoot_values = gamma.pdf(times, 12)
    values = peak_values - 0.35 * undershoot_values
    hrf_at_trs = values / np.max(values) * 0.6

    return hrf_at_trs

def event_to_hrf(event_series, hrf_at_trs):
    """ Convolve an HRF to movie event data """

    convolved = np.convolve(event_series, hrf_at_trs)
    n_to_remove = len(hrf_at_trs) - 1
    convolved = convolved[:-n_to_remove]

    return convolved
