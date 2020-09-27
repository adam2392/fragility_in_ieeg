import os
import shutil
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
from natsort import natsorted
from tqdm import tqdm

from eztrack.base.metrics import counted
from eztrack.base.utils.errors import EZTrackValueError, EZTrackOSError
from eztrack.utils.config import logger as logger
from eztrack.pipeline.analysis.execute.basepipe import BasePipe


def _get_tempfileindex(x):
    buff = x.split("_")
    winnum = int(buff[-1].split(".")[0])
    return winnum


class RunMergeModel(BasePipe):
    """
    Pipeline class for merging data that is computed on separate (possibly overlapping) windows of data.

    It is a post-merging step after parallelized processing of data. It merges together either:
        1. ltvn model
        2. min-2norm perturbation model

    Attributes
    ----------
    tempdir : os.PathLike
        Window size of the data that is passed in.
    numwins : int
        Step size of the data that will be generated

    Notes
    -----
    Depends on the file name pattern that is used in saving algorithm output.

    Examples
    --------
    >>> import os
    >>> from eztrack.pipeline.analysis.execute.runmerge import RunMergeModel
    >>> model_params = {
    ...     'tempdir': os.path.join(),
    ...     'numwins': 2459,
    ...     }
    >>> modelmerger = RunMergeModel(**model_params)
    >>> modelmerger,mergefragilitydata(output_fname="")
    """

    def __init__(self):
        super(RunMergeModel, self).__init__()
        # store number of windows to analyze
        self.numwins = numwins
        self.output_fpath = None

        # get the list of all the files in natural order
        alltempfiles = [
            f
            for f in os.listdir(self.tempdir)
            if f.endswith(".npz")
            if not f.startswith(".")
        ]
        self.temp_fpaths = natsorted(alltempfiles)
        self._store_params()

    def _store_params(self, **kwargs):
        self.parameter_dict.update(numwins=self.numwins)
        self.parameter_dict.update(**kwargs)

    def _clean_file_path(self, temp_fpath, outputfilename):
        # remove that tempfilename
        os.remove(temp_fpath)
        print("File Removed {}!".format(temp_fpath))

        # remove success file
        outfile = os.path.basename(outputfilename)
        flag_fpath = outfile.replace("fragmodel.npz", "frag_success.txt")
        main_dir = Path(self.tempdir).parent.parent.as_posix()
        success_flag_name = os.path.join(main_dir, flag_fpath)
        try:
            os.remove(success_flag_name)
            print("File Removed {}!".format(success_flag_name))
        except Exception as e:
            print(e)
            print("Success flag file already removed!")

    @property
    def numcompletedfiles(self):
        """Count of temporarily saved filepaths."""
        return len(self.temp_fpaths)

    def load_data(self, metadata: Dict, kwargs: Dict = None) -> Dict:
        """Load data, so that fit() can be called."""
        if kwargs is None:
            kwargs = dict()
        self.parameter_dict.update(**kwargs)
        self.parameter_dict.update(metadata)
        return self.parameter_dict

    def fit(self, output_fname: str, temp_fpaths: List = []):
        """
        Merge computed data from separate files into 1 zipped numpy compressed file.

        Parameters
        ----------
        output_fname : str
            output filename that will be saved that contains the merged results.
        temp_fpaths : List
            temporary filepaths that are being written.

        Returns
        -------
        pertmats, adjmats, delvecs_array : np.ndarray, np.ndarray, np.ndarray
            The minimum norms of all channels across time (C x T)

            The tensor of state matrices over time (C x C x T) (The A matrix for time point 0, is [...,0])

            The delta vectors for each channel across time (C x C x T) ([0,:,0] is the delta vector for the first channel
            at time point 0).

        """
        # initialize a flag to save forsure, unless errors occur
        errors_list = []

        if len(temp_fpaths) != len(self.temp_fpaths):
            warnings.warn(
                "Temporary file path should be set to the same number of "
                "temp_fpaths found."
            )
        else:
            self.temp_fpaths = temp_fpaths

        # zip files over windows of analyzed data
        for idx, temp_fpath in enumerate(tqdm(self.temp_fpaths)):
            # get the window numbr of this file just to check
            tempfile_index = _get_tempfileindex(temp_fpath)
            if tempfile_index != idx:
                logger.exception(f"Win num {tempfile_index} should match idx {idx}")
                counted("Backend_exception")
                raise EZTrackValueError(
                    f"Win num {tempfile_index} should match idx {idx}"
                )
            if not os.path.exists(temp_fpath):
                temp_fpath = os.path.join(self.tempdir, temp_fpath)

            # load result data
            try:
                data_struct = np.load(temp_fpath)
            except OSError as e:
                counted("Backend_exception")
                EZTrackOSError(e[1])
                self._clean_file_path(temp_fpath, output_fname)
                errors_list.append(e)

            # extract the data elements from dictionary
            pertmat = data_struct["pertmat"]
            delvecs = data_struct["delvecs"]
            adjmat = data_struct["adjmat"]

            if idx == 0:
                n_chs = len(pertmat.squeeze())
                n_wins = len(self.temp_fpaths)

                # initialize arrays
                pertmats = np.zeros((n_chs, n_wins))
                delvecs_array = np.zeros((n_chs, n_chs, n_wins))
                adjmats = np.zeros((n_chs, n_chs, n_wins))

            # store results
            pertmats[:, idx] = pertmat.squeeze()
            delvecs_array[..., idx] = delvecs
            adjmats[..., idx] = adjmat

        # store data in RAM of object
        self.pertmats = pertmats
        self.adjmats = adjmats
        self.delvecs_array = delvecs_array

        if len(errors_list) == 0:
            # save adjmats, pertmats and delvecs array along with metadata
            np.savez_compressed(
                output_fname,
                adjmats=adjmats,
                pertmats=pertmats,
                delvecs_arr=delvecs_array,
            )
            self.output_fpath = output_fname
            self._store_params(output_fpath=self.output_fpath)

            # if successful, remove temporary files
            shutil.rmtree(self.tempdir)

        return np.array(pertmats), np.array(adjmats), np.array(delvecs_array)
