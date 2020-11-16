import datetime
import os
import warnings
from abc import abstractmethod

import numpy as np
from natsort import natsorted

from eztrack.base.metrics import counted
from eztrack.utils.data_structures_utils import _ensure_list
from eztrack.base.utils.errors import EZTrackAttributeError
from eztrack.utils.config import logger as logger
from eztrack.fragility.basemodel import BaseWindowModel


class BasePipe(BaseWindowModel):
    """
    Base executing class for fragility module.

    Makes sure children classes inherit certain function names.

    Attributes
    ----------
    winsize : int
        Window size of the data that is passed in.
    stepsize : int
        Step size of the data that will be generated
    samplerate : int
        Number of dimensions (this is always 2)
    """

    def __init__(
        self,
        winsize: int = None,
        stepsize: int = None,
        samplerate: float = None,
        tempdir: os.PathLike = None,
    ):
        super(BasePipe, self).__init__()

        if samplerate != None:
            if samplerate < 200:
                logger.warning(
                    f"Sample rate is {samplerate} (< 200), which is difficult to say works!"
                )
                warnings.warn("Sample rate is < 200, which is difficult to say works!")
        if winsize != None:
            if not isinstance(winsize, int):
                logger.exception(
                    f"window size should be passed as an integer! You passed {winsize}."
                )
                counted("Backend_exception")
                raise EZTrackAttributeError(
                    f"window size should be passed as an integer! You passed {winsize}."
                )
        if stepsize != None:
            if not isinstance(stepsize, int):
                logger.exception(
                    f"Step size should be passed as an integer! You passed {stepsize}."
                )
                counted("Backend_exception")
                raise EZTrackAttributeError(
                    f"Step size should be passed as an integer! You passed {stepsize}."
                )

        # log start time
        self.start_time = datetime.datetime.utcnow()

        # self.model = model
        self.winsize = winsize
        self.stepsize = stepsize
        self.samplerate = samplerate
        self.tempdir = tempdir

        # compute the number of samples in window and step
        self._setsampsinwin()
        self._setsampsinstep()

    def get_metadata(self):
        """Get parameter metadata dictionary."""
        return self.parameter_dict

    @property
    def winsize_ms(self) -> float:
        """Window size in milliseconds."""
        return np.divide(self.winsamps, self.samplerate)

    @property
    def stepsize_ms(self) -> float:
        """Step size in milliseconds."""
        return np.divide(self.stepsamps, self.samplerate)

    @property
    def winsize_samples(self) -> int:
        """Window size in number of samples."""
        return self.winsamps

    @property
    def stepsize_samples(self) -> int:
        """Step size in number of samples."""
        return self.stepsamps

    def _setsampsinwin(self):
        self.winsamps = self.winsize
        if self.winsamps != None:
            if self.winsamps % 1 != 0:
                logger.warning(
                    f"The number of samples within the window size is {self.winsamps} which is not an "
                    "even integer. Consider increasing/changing the window size."
                )
                warnings.warn(
                    "The number of samples within your window size is not an even integer. "
                    "Consider increasing/changing the window size."
                )

    def _setsampsinstep(self):
        self.stepsamps = self.stepsize
        if self.stepsamps != None:
            if self.stepsamps % 1 != 0:
                logger.warning(
                    f"The number of samples within the window size is {self.winsamps} which is not an "
                    "even integer. Consider increasing/changing the window size."
                )
                warnings.warn(
                    "The number of samples within your step size is not an even integer. "
                    "Consider increasing/changing the step size."
                )

    def compute_timepoints(self):
        """
        Compute the corresponding timepoints of each window in terms of seconds.

        :return: timepoints (list) a list of the timepoints in seconds of each window begin/end.
        """
        timepoints = self.samplepoints / self.samplerate * 1000
        self.timepoints = timepoints
        return timepoints

    def compute_samplepoints(self, numtimepoints, copy=True):
        """
        Compute the index endpoints in terms of signal samples.

        Does it for each sliding window in a piped algorithm.

        :param numtimepoints: (int) T in an NxT matrix of raw data.
        :param copy: (bool) should the function return a copy?
        :return: samplepoints (list; optional) list of samplepoint indices that define each window
        of data
        """
        # Creates a [n,2] array that holds the sample range of each window that
        # is used to index the raw data for a sliding window analysis
        samplestarts = np.arange(
            0, numtimepoints - self.winsize + 1.0, self.stepsize
        ).astype(int)
        sampleends = np.arange(self.winsize - 1.0, numtimepoints, self.stepsize).astype(
            int
        )
        samplepoints = np.append(
            samplestarts[:, np.newaxis], sampleends[:, np.newaxis], axis=1
        )
        self.numwins = samplepoints.shape[0]
        if copy:
            self.samplepoints = samplepoints
        else:
            return samplepoints

    def getmissingwins(self, tempresultsdir, numwins):  # pragma: no cover
        """
        Compute the missing data within the temporary local directory.

        Assumes files are separated by "_" with the second part being the number of the
        windowed result. E.g. temp_0.npz, ..., temp_49.npz, temp_51.npz, temp_100.npz
        will have windows {1,...,100} there, but missing temp_50.npz

        :param tempresultsdir: (os.PathLike) where to check for temporary result files.
        :param numwins: (int) the number of windows that is needed to complete the
                            models through the time series data
        :return: winstoanalyze (list) is the list of window indices to analyze
        """
        # get a list of all the sorted files in our temporary directory
        tempfiles = [f for f in os.listdir(tempresultsdir) if not f.startswith(".")]
        tempfiles = natsorted(tempfiles)

        if len(tempfiles) == 0:
            return np.arange(0, numwins).astype(int)

        if numwins != len(tempfiles):
            # if numwins does not match, get list of wins not completed
            totalwins = np.arange(0, numwins, dtype="int")
            tempfiles = np.array(tempfiles)[:, np.newaxis]

            # patient+'_'+str(iwin) = the way files are named
            def func(filename):
                return int(filename[0].split("_")[-1].split(".")[0])

            tempwins = np.apply_along_axis(func, 1, tempfiles)
            winstoanalyze = list(set(totalwins) - set(tempwins))
        else:
            winstoanalyze = []
        return _ensure_list(winstoanalyze)

    @staticmethod
    def _get_tempfilename(x):
        return "temp_{}.npz".format(x)

    @abstractmethod
    def fit(self, X):  # pragma: no cover
        """Scikit-learn style method for transforming data pipelines."""
        pass
