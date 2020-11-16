import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Union

import numpy as np
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

import eztrack.base.dataconfig as constants
from eztrack.base.metrics import counted
from eztrack.base.utils.errors import EZTrackAttributeError, EZTrackRuntimeError
from eztrack.utils.config import logger as logger
from eztrack.fragility.mvarmodel import MvarModel
from eztrack.fragility.perturbationmodel import MinNormPerturbModel
from eztrack.pipeline.analysis.execute.basepipe import BasePipe


# @memory.cache
def _compute_fragility_func(
    shared_mvarmodel, shared_pertmodel, raw_data, samplepoints, tempdir, win
):  # pragma: no cover
    # 1: fill matrix of all channels' next EEG data over window
    win_begin = samplepoints[win, 0]
    win_end = samplepoints[win, 1] + 1
    eegwin = raw_data[:, win_begin:win_end]

    # 2: compute state transition matrix using mvar model
    adjmat = shared_mvarmodel.fit(eegwin)

    # 3: compute perturbation model
    pertmat = shared_pertmodel.fit(adjmat)
    delvecs = shared_pertmodel.minimum_delta_vectors

    # save adjacency matrix
    tempfilename = os.path.join(tempdir, BasePipe._get_tempfilename(win))
    try:
        np.savez(tempfilename, adjmat=adjmat, pertmat=pertmat, delvecs=delvecs)
    except BaseException:
        return (None, None, None)
    return pertmat, adjmat, delvecs


class RunFragModel(BasePipe):
    """
    A Pipeline class for running the fragility model on data.

    It implements multiprocessing or window based parallelization,
    so that users may easily deploy algorithm on multi-CPU.

    Implements the linear time-varying network model (of a multivariate autoregressive nature p=1)
    and a minimum-2norm structured perturbation model and fragility metric column normalization.

    Attributes
    ----------
    winsize : int
        Window size of the data that is passed in.
    stepsize : int
        Step size of the data that will be generated
    sfreq : int
        Number of dimensions (this is always 2)

    Notes
    -----
    When the size of the data is too large (e.g. N > 180, W > 1000), then right now the construction of the csr
    matrix scales up. With more efficient indexing, we can perhaps decrease this.

    Examples
    --------
    >>> import numpy as np
    >>> from eztrack.pipeline.analysis.execute.runfrag import RunFragModel
    >>> model_params = {
    ...     'winsize': 250,
    ...     'stepsize': 125,
    ...     'sfreq': 1000,
    ...     'radius': 1.5,
    ...     'perturb_type': 'C',
    ...     'numcores': 48,
    ...     }
    >>> modelrunner = RunFragModel(**model_params)
    >>> data = np.random.rand((80,5000))
    >>> # load in the data for modelrunner
    >>> modelrunner.load_data(data)
    >>> for iwin in range(len(modelrunner.samplepoints)):
    ...     samplewin = samplepoints[iwin]
    ...     A, pertnorm, delvec = modelrunner.runwindow(iwin)
    ...     print(A.shape, pertnorm, delvec.shape)
    """

    def __init__(
        self,
        winsize: int = constants.WINSIZE_LTV,
        stepsize: int = constants.STEPSIZE_LTV,
        sfreq: float = None,
        radius: float = constants.RADIUS,
        perturb_type: str = constants.PERTURBTYPE,
        numcores: int = None,
        tempdir: os.PathLike = tempfile.TemporaryDirectory(),
        searchnum: int = None,
        l2penalty: float = 0,
        eigs_quality_check: bool = False,
        stabilizeflag: bool = False,
        apply_orthogonal_constraint: bool = False,
        method_to_use: str = "hankel",
        svd_rank: float = 1.0,
    ):
        super(RunFragModel, self).__init__(winsize, stepsize, sfreq, tempdir)

        # initializes the implicit data attributes to None
        self.adjmats = None
        self.pertmats = None
        self.delvecs = None

        self.rawdata = None

        # main parameters that need to be set
        self.winsize = winsize
        self.stepsize = stepsize
        self.samplerate = sfreq
        self.radius = radius
        self.perturbtype = perturb_type
        self.numcores = numcores

        # get additional kwargs
        self.searchnum = searchnum
        self.l2penalty = l2penalty
        self.stabilizeflag = stabilizeflag
        self.eigs_quality_check = eigs_quality_check
        self.apply_orthogonal_constraint = apply_orthogonal_constraint
        self.method_to_use = method_to_use
        self.svd_rank = svd_rank
        # separate kwargs into ltvkwargs and perturbkwargs
        ltvmodel_kwargs = {
            "stabilize": self.stabilizeflag,
            "l2penalty": self.l2penalty,
            "method_to_use": self.method_to_use,
            "maxeig": 1.0,
            "multitaper": False,
            "svd_rank": self.svd_rank,
        }
        self.mvarmodel = MvarModel(**ltvmodel_kwargs)

        pertmodel_kwargs = {
            "radius": self.radius,
            "perturb_type": self.perturbtype,
            "eigs_quality_check": self.eigs_quality_check,
            "apply_orthogonal_constraint": self.apply_orthogonal_constraint,
        }
        self.pertmodel = MinNormPerturbModel(**pertmodel_kwargs)

        # store all the parameters
        allkwargs = ltvmodel_kwargs
        allkwargs.update(pertmodel_kwargs)
        self._store_params(**allkwargs)
        self._initialize_cores()

    def _initialize_cores(self):
        if self.numcores == None:
            self.numcores = cpu_count() // 2
        if self.numcores > cpu_count():
            self.numcores = cpu_count() // 2

    def _store_params(self, **kwargs):
        self.parameter_dict.update(**kwargs)

    def load_data(self, rawdata):
        """
        Load in raw data that is NxT.

        Data will be ran through with the fragility algorithm.

        Parameters
        ----------
        rawdata : (np.ndarray) NxT matrix that is N contacts and T signal samples.
        """
        self.rawdata = rawdata
        # get number of channels and samples in the raw data
        self.numchans, self.numsignals = self.rawdata.shape
        # compute time and sample windows array
        self.compute_samplepoints(self.numsignals)
        self.compute_timepoints()
        self.numwins = self.samplepoints.shape[0]

        # self.logger.info("Loaded in raw data to begin fragility computation: shape = {}, "
        #                  "number of windows = {}".format(self.rawdata.shape, len(self.samplepoints)))

    def fit(self, parallel=True, output_fpath=None):
        """Run all windows through fragility algo."""
        if self.rawdata is None:
            logger.exception("Attempted to run fit() before loading data.")
            counted("Backend_exception")
            raise EZTrackRuntimeError("Before running fit(), please load_data(data).")

        if parallel:
            results = self.runparallel(
                compute_on_missing_wins=True, output_fpath=output_fpath
            )
        else:
            pertmats = np.zeros((self.numchans, self.numwins))
            adjmats = np.zeros((self.numchans, self.numchans, self.numwins))
            delvecs_arr = np.zeros((self.numchans, self.numchans, self.numwins))
            for i in tqdm(range(self.numwins), desc="windows run"):
                pertmat, adjmat, delvecs = self.runwindow(i)
                pertmats[:, i] = pertmat
                adjmats[..., i] = adjmat
                delvecs_arr[..., i] = delvecs
            results = (pertmats, adjmats, delvecs_arr)

        return results

    def runwindow(self, iwin, fast=True, save=True):
        """
        Run a specific window of data that is defined by self.samplepoints.

        Parameters
        ----------
        iwin : (int)
        normalize : (bool)
        fast : (bool)
        save : (bool)

        Returns
        -------
                adjmat (np.ndarray)
                pertnorm (float)
                delvec (np.ndarray)
        """
        assert self.numchans <= self.numsignals
        if save:
            if self.tempdir is None:
                logger.error(
                    "You are trying to save resulting computation, \
                                 but don't have tempdir set."
                )
                counted("Backend_exception")
                raise EZTrackAttributeError(
                    "You are trying to save resulting computation, \
                    but don't have tempdir set."
                )

        if self.numchans >= 200:
            logger.warning(
                f"You are attempting to run a file with {self.numchans} channels. We recommend a maximum"
                f"of 200 channels. Expect a longer runtime."
            )

        # 1: fill matrix of all channels' next EEG data over window
        win_begin = self.samplepoints[iwin, 0]
        win_end = self.samplepoints[iwin, 1] + 1
        eegwin = self.rawdata[:, win_begin:win_end]

        # 2. Compute the mvar-1 model
        self.adjmat = self.mvarmodel.fit(eegwin).squeeze()

        # initialize dict
        perturbation_dict = dict()

        # perform perturbation model computation
        if not fast:
            self.pertmat = self.pertmodel._compute_gridsearch_perturbation(
                self.adjmat, searchnum=self.searchnum
            )
            self.delvecs = self.pertmodel.minimum_delta_vectors
            self.delfreqs = self.pertmodel.minimum_freqs
            # save the corresponding arrays
            perturbation_dict["pertmat"] = self.pertmat
            perturbation_dict["delvecs"] = self.delvecs
            perturbation_dict["delfreqs"] = self.delfreqs
        else:
            self.pertmat = self.pertmodel.fit(self.adjmat)
            self.delvecs = self.pertmodel.minimum_delta_vectors

            # save the corresponding arrays
            perturbation_dict["pertmat"] = self.pertmat
            perturbation_dict["delvecs"] = self.delvecs

        if save:
            tempfilename = os.path.join(self.tempdir, BasePipe._get_tempfilename(iwin))
            np.savez_compressed(
                tempfilename,
                adjmat=self.adjmat,
                pertmat=self.pertmat,
                delvecs=self.delvecs,
            )
        return self.pertmat, self.adjmat, self.delvecs

    def runparallel(self, compute_on_missing_wins, output_fpath: Union[str, Path]):
        """
        Run parallelized fragility algorithm.

        Parameters
        ----------
        compute_on_missing_wins :
        output_fpath :

        Returns
        -------
        pertmats : np.ndarray
            C x T array.
        adjmats : np.ndarray
            C x C x T array.
        delvecs_arr : np.ndarray

        """
        # determine missing windows in temporary directory
        if compute_on_missing_wins:
            window_inds = self.getmissingwins(self.tempdir, self.numwins)
        else:
            window_inds = range(self.numwins)

        # run parallelized job to compute fragility over all windows
        start = time.time()
        fragility_results = Parallel(n_jobs=self.numcores)(
            delayed(_compute_fragility_func)(
                self.mvarmodel,
                self.pertmodel,
                self.rawdata,
                self.samplepoints,
                self.tempdir,
                win,
            )
            for win in tqdm(window_inds)
        )
        stop = time.time()
        print("Elapsed time for the entire processing: {:.2f} s".format(stop - start))

        # initialize numpy arrays to return results
        pertmats = np.zeros((self.numchans, self.numwins))
        adjmats = np.zeros((self.numchans, self.numchans, self.numwins))
        delvecs_arr = np.zeros(
            (self.numchans, self.numchans, self.numwins), dtype=np.complex
        )
        for i in range(len(fragility_results)):
            pertmat, adjmat, delvecs = fragility_results[i]
            pertmats[:, i] = pertmat
            adjmats[..., i] = adjmat
            delvecs_arr[..., i] = delvecs

        # save these results and remove cached directory
        if output_fpath is not None:
            np.savez_compressed(
                output_fpath,
                adjmats=adjmats,
                pertmats=pertmats,
                delvecs_arr=delvecs_arr,
            )
            # if successful, remove temporary files
            shutil.rmtree(self.tempdir, ignore_errors=True)
        return pertmats, adjmats, delvecs_arr
