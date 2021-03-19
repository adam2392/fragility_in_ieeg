import json
from datetime import date, datetime
from typing import Tuple, Dict

import mne
import numpy as np
import scipy
from mne.utils import warn


def _apply_threshold(X, threshold, default_val=0.0):
    X = X.copy()
    X[X < threshold] = default_val
    return X


def _smooth_vector(x, window_len=5, window="hanning"):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    Parameters
    ----------
    x : the input signal
    window_len : the dimension of the smoothing window; should be an odd integer
    window : the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns
    -------
    y : the smoothed signal

    Examples
    --------
    >>> from eztrack.base.utils.preprocess_utils import _smooth_vector
    >>> t=linspace(-2,2,0.1)
    >>> x=sin(t)+randn(len(t))*0.1
    >>> y=smooth(x)

    See also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len < 3:
        return x

    s = np.r_[x[window_len - 1: 0: -1], x, x[-2: -window_len - 1: -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


def _smooth_matrix(X, axis=0, window_len=5, window="hanning"):
    if X.ndim > 2:
        raise RuntimeError("Matrix X should be only 2-dimensional.")

    if axis == 0:
        iternum = X.shape[0]
    elif axis == 1:
        iternum = X.shape[1]

    X_smooth = []
    for i in range(iternum):
        x_vec = X[i, :].squeeze()
        x_smooth_vec = _smooth_vector(x_vec, window_len, window)

        # append to matrix
        X_smooth.append(x_smooth_vec)
    X_smooth = np.array(X_smooth)
    return X_smooth


def _resample_seizure(mat, onset_window, offset_window, desired_length=500):
    pre_mat = mat[:, :onset_window]
    ictal_mat = mat[:, onset_window:offset_window]
    post_mat = mat[:, offset_window:]

    ictal_mat = _resample_mat(ictal_mat, desired_length)
    return np.concatenate((pre_mat, ictal_mat, post_mat), axis=1)


def _resample_mat(mat, desired_len):
    """
    Resample an entire matrix composed of signals x time.

    Resamples each signal, one at a time.

    Parameters
    ----------
    mat
    desired_len

    Returns
    -------

    """
    if mat.ndim != 2:
        raise ValueError("Matrix needs to be 2D.")

    # initialize resampled matrix
    resampled_mat = np.zeros((mat.shape[0], desired_len))

    # resample each vector
    for idx in range(mat.shape[0]):
        seq = mat[idx, ...].squeeze()
        resampled_mat[idx, :] = scipy.signal.resample(seq, desired_len)
    return resampled_mat


def _map_events_to_window(
        raw: mne.io.BaseRaw, winsize: int, stepsize: int
) -> (np.ndarray, Dict):
    """Map events/events_id to window based sampling."""
    # get events and convert to annotations
    events, events_id = mne.events_from_annotations(raw, event_id=None, verbose=False)

    # get the length of recording
    length_recording = len(raw)

    # compute list of end-point windows for analysis
    samplepoints = _compute_samplepoints(winsize, stepsize, length_recording)

    # map each event onset to a window
    for i in range(events.shape[0]):
        event_onset_sample = events[i, 0]
        # print(event_onset_sample)
        # print(samplepoints)
        event_onset_window = _sample_to_window(event_onset_sample, samplepoints)
        events[i, 0] = event_onset_window

    return events, events_id


def _sample_to_window(sample, samplepoints):
    return int(
        np.where((samplepoints[:, 0] <= sample) & (samplepoints[:, 1] >= sample))[0][0]
    )


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    Pass to json.dump(), or json.load().
    """

    def default(self, obj):  # noqa
        if isinstance(
                obj,
                (
                        np.int_,
                        np.intc,
                        np.intp,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def _get_onset_event_id(events, event_id):
    if event_id is None:
        return None

    # cast to numpy array
    events = np.asarray(events)

    if event_id not in events[:, 2]:
        warn(f"{event_id} event ID is not inside the events data structure.")
        return None

    event_ind = np.where(events[:, 2] == event_id)[0][0]
    return events[event_ind, 0]


def _compute_samplepoints(winsamps, stepsamps, numtimepoints):
    # Creates a [n,2] array that holds the sample range of each window that
    # is used to index the raw data for a sliding window analysis
    samplestarts = np.arange(0, numtimepoints - winsamps + 1.0, stepsamps).astype(int)
    sampleends = np.arange(winsamps, numtimepoints + 1, stepsamps).astype(int)

    samplepoints = np.append(
        samplestarts[:, np.newaxis], sampleends[:, np.newaxis], axis=1
    )
    return samplepoints


def _select_window(X: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    """Select a window slice in X."""
    if window[0] < 0 or window[1] > X.shape[1]:
        raise RuntimeError(
            f"Window needs to be range between [0, T], "
            f"where T is the length of data matrix X ({X.shape})."
        )
    return X[:, window[0]: window[1]]


class Normalize:
    """Class of normalization methods to apply to NxT matrices."""

    @staticmethod
    def compute_fragilitymetric(minnormpertmat, invert=False):
        """
        Normalization of a NxT matrix to [0,1).

        Normalizes what we defined as the fragility metric. It emphasizes
        values over the columns that is significantly different from the
        lowest value, normalized by the range of the entire column.

        This is an unsymmetric normalization transformation.

        Parameters
        ----------
        minnormpertmat :

        Returns
        -------
        fragilitymat :
        """
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape

        # assert N < T
        fragilitymat = np.zeros((N, T))
        for icol in range(T):
            if invert:
                # minnormpertmat = -1 * minnormpertmat

                numerator_rel_range = minnormpertmat[:, icol] - np.min(
                    minnormpertmat[:, icol]
                )
                denominator = np.max(minnormpertmat[:, icol])
            else:
                numerator_rel_range = (
                        np.max(minnormpertmat[:, icol]) - minnormpertmat[:, icol]
                )
                denominator = np.max(minnormpertmat[:, icol])
            fragilitymat[:, icol] = numerator_rel_range / denominator
        return fragilitymat

    @staticmethod
    def compute_minmaxfragilitymetric(minnormpertmat):
        """
        Min-Max normalization of a NxT matrix to [0,1].

        It emphasizes values over the columns that is
        significantly different from the lowest value,
        normalized by the range of the entire column. It maps each
        column to values with 0 and 1 definitely.

        Parameters
        ----------
        minnormpertmat :

        Returns
        -------
        fragilitymat :
        """
        import numpy.matlib

        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape

        # get the min/max for each column in matrix
        minacrosstime = np.min(minnormpertmat, axis=0)
        maxacrosstime = np.max(minnormpertmat, axis=0)

        # normalized data with minmax scaling
        fragilitymat = -1 * np.true_divide(
            (minnormpertmat - np.matlib.repmat(maxacrosstime, N, 1)),
            np.matlib.repmat(maxacrosstime - minacrosstime, N, 1),
        )
        return fragilitymat

    @staticmethod
    def compute_znormalized_fragilitymetric(minnormpertmat):
        """
        Z-normalization of each column of a NxT matrix.

        Parameters
        ----------
        minnormpertmat :

        Returns
        -------
        fragmat :
        """
        # get mean, std
        avg_contacts = np.mean(minnormpertmat, keepdims=True, axis=1)
        std_contacts = np.std(minnormpertmat, keepdims=True, axis=1)

        # normalized data with minmax scaling
        return (minnormpertmat - avg_contacts) / std_contacts


def _subsample_matrices_in_time(mat_list):
    maxlen = min([x.shape[1] for x in mat_list])
    if maxlen < 50:
        raise RuntimeError("Preferably not under 50 samples...")

    mat_list = [x[:, :maxlen] for x in mat_list]
    return mat_list
