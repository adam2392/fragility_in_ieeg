import copy as cp
import json
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Union, Dict

import bids
import mne
import numpy as np
from mne.channels.channels import ContainsMixin
from mne.externals.h5io import write_hdf5
from mne.io import RawArray
from mne.io.pick import pick_channels
from mne.time_frequency.csd import (
    _vector_to_sym_mat,
    _n_dims_from_triu,
    _sym_mat_to_vector,
)
from mne_bids.path import _parse_ext

from analysis.publication.utils import (Normalize)


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


class DERIVATIVETYPES(Enum):
    STATE_MATRIX = "statematrix"
    PERTURB_MATRIX = "perturbmatrix"
    DELTAVECS_MATRIX = "deltavecsmatrix"


def pick_channels_connectivity(csd, include=[], exclude=[], ordered=False, copy=True):
    """Pick channels from connectivity matrix.
    Parameters
    ----------
    csd : instance of ResultConnectivity
        The Result object to select the channels from.
    include : list of str
        List of channels to include (if empty, include all available).
    exclude : list of str
        Channels to exclude (if empty, do not exclude any).
    ordered : bool
        If True (default False), ensure that the order of the channels in the
        modified instance matches the order of ``include``.
        .. versionadded:: 0.20.0
    copy : bool
        If True (the default), return a copy of the CSD matrix with the
        modified channels. If False, channels are modified in-place.
        .. versionadded:: 0.20.0
    Returns
    -------
    res : instance of CrossSpectralDensity
        Cross-spectral density restricted to selected channels.
    """
    if copy:
        csd = csd.copy()

    sel = pick_channels(csd.ch_names, include=include, exclude=exclude, ordered=ordered)
    data = []
    for vec in csd._data.T:
        mat = _vector_to_sym_mat(vec)
        mat = mat[sel, :][:, sel]
        data.append(_sym_mat_to_vector(mat))
    ch_names = [csd.ch_names[i] for i in sel]

    csd._data = np.array(data).T
    csd.ch_names = ch_names
    return csd


class ResultConnectivity(ContainsMixin):
    """Cross connectivity (spectral/correlation) matrices.
    Largely taken from :func:`mne.time_frequency.CrossSpectralDensity`.
    Given a list of time series, the connectivity matrix denotes for each pair of time
    series, the cross-spectral density, or cross-correlation.
    This matrix is symmetric and internally stored as a vector.
    This object can store multiple CSD matrices: one for each frequency.
    Use ``.get_data(freq)`` to obtain an CSD matrix as an ndarray.
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_channels, times)
        For each frequency, the cross-spectral density matrix in vector format.
    info : Info
        The measurement info.
    metadata : ResultInfo
        The result metadata info.
    comment : str | None, default None
        Comment on the data, e.g., the experimental condition.
    method : str | None, default None
        Comment on the method used to compute the data, e.g., morlet wavelet.
    fmin: float | None, default None
        The lower end of frequency
    fmax : float | None, default None
        The upper end of frequency
    %(verbose)s
    """

    def __init__(
            self,
            data,
            info,
            metadata=None,
            comment=None,
            method=None,
            fmin=None,
            fmax=None,
    ):
        if data.ndim != 3:
            raise ValueError("data should be 3d. Got %d." % data.ndim)
        n_channels, _, n_times = data.shape

        if n_channels != len(info["chs"]):
            raise ValueError(
                "Number of channels and data size don't match"
                " (%d != %d)." % (n_channels, len(info["chs"]))
            )
        if fmin is not None and fmax is not None:
            if fmin > fmax:
                raise ValueError(
                    f"'fmin' ({fmin}) should not be " f"greater then 'fmax' ({fmax})"
                )

        self.data = data
        self.fmin = fmin
        self.fmax = fmax
        self.comment = comment
        self.method = method
        self.preload = True

        data = np.asarray(data)
        if data.ndim == 3:
            _data = []
            for itime in range(data.shape[-1]):
                _data.append(_sym_mat_to_vector(data[..., itime]))
            data = np.asarray(_data).T
        elif data.ndim != 3:
            raise ValueError("`data` should be a 3D array.")
        self._data = data

        ch_names = info["ch_names"]
        if len(ch_names) != _n_dims_from_triu(data.shape[0]):
            raise ValueError(
                "Number of ch_names does not match the number of "
                f"time series in the CSD matrix {_n_dims_from_triu(len(data))}."
            )
        self.info = ResultInfo(source_info=info, **metadata)
        self.times = np.arange(self.n_times) / self.metadata["model_params"]["stepsize"]

    @property
    def metadata(self):
        return self.info["result_info"]

    def _update_times(self):
        """Update times."""
        self._times = np.arange(self.n_times) / float(self.info["sfreq"])
        # make it immutable
        self._times.flags.writeable = False

    @property
    def n_times(self):
        """Number of time points."""
        return self._data.shape[-1]

    @property
    def ch_names(self):
        """Channel names."""
        return self.info["ch_names"]

    @property
    def n_channels(self):
        """Number of time series defined in this CSD object."""
        return len(self.ch_names)

    def __len__(self):  # noqa: D105
        """Return number of frequencies.
        Returns
        -------
        n_freqs : int
            The number of frequencies.
        """
        return len(self.times)

    def __repr__(self):  # noqa: D105
        # Make a pretty string representation of the frequencies
        if self.fmin or self.fmax:
            freq_str = ", ".join([self.fmin, self.fmax]) + " Hz."
        else:
            freq_str = ""

        time_str = len(self)

        return (
            "<ConnectivityMatrix  |  " "n_channels={}, time={}, frequencies={}>"
        ).format(self.n_channels, time_str, freq_str)

    def get_data(self, start=0, stop=None):
        """Get the CSD matrix for a given frequency as NumPy array.
        If there is only one matrix defined in the CSD object, calling this
        method without any parameters will return it. If multiple matrices are
        defined, use either the ``frequency`` or ``index`` parameter to select
        one.
        Parameters
        ----------
        start : int | 0
            Return the CSD matrix for the time point
        stop : int | 0
            Return the CSD matrix for the time point
        Returns
        -------
        csd : ndarray, shape (n_channels, n_channels, n_times)
            The CSD matrix corresponding to the requested frequency.
        """
        if stop is None:
            stop = self.n_times
        return _vector_to_sym_mat(self._data[..., start:stop])

    def __setstate__(self, state):  # noqa: D105
        self._data = state["data"]
        self.tmin = state["tmin"]
        self.tmax = state["tmax"]
        self.frequencies = state["frequencies"]
        self.n_fft = state["n_fft"]

    def __getstate__(self):  # noqa: D105
        return dict(
            data=self._data,
            tmin=self.tmin,
            tmax=self.tmax,
            ch_names=self.ch_names,
            frequencies=self.frequencies,
            n_fft=self.n_fft,
        )

    def __getitem__(self, sel):  # noqa: D105
        """Subselect frequencies.
        Parameters
        ----------
        sel : ndarray
            Array of frequency indices to subselect.
        Returns
        -------
        csd : instance of CrossSpectralDensity
            A new CSD instance with the subset of frequencies.
        """
        return ResultConnectivity(
            data=_vector_to_sym_mat(self._data[..., sel]),
            info=self.info,
            metadata=self.metadata,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def save(self, fname):
        """Save the CSD to an HDF5 file.
        Parameters
        ----------
        fname : str
            The name of the file to save the CSD to. The extension '.h5' will
            be appended if the given filename doesn't have it already.
        See Also
        --------
        read_csd : For reading CSD objects from a file.
        """
        if not fname.endswith(".h5"):
            fname += ".h5"

        write_hdf5(fname, self.__getstate__(), overwrite=True, title="conpy")

    def copy(self):
        """Return copy of the CrossSpectralDensity object.
        Returns
        -------
        copy : instance of CrossSpectralDensity
            A copy of the object.
        """
        return cp.deepcopy(self)

    def pick_channels(self, ch_names, ordered=False):
        """Pick channels from this cross-spectral density matrix.
        Parameters
        ----------
        ch_names : list of str
            List of channels to keep. All other channels are dropped.
        ordered : bool
            If True (default False), ensure that the order of the channels
            matches the order of ``ch_names``.
        Returns
        -------
        csd : instance of CrossSpectralDensity.
            The modified cross-spectral density object.
        Notes
        -----
        Operates in-place.
        .. versionadded:: 0.20.0
        """
        return pick_channels_connectivity(
            self, include=ch_names, exclude=[], ordered=ordered, copy=False
        )


class ResultInfo(mne.Info):
    """Result information.
    This data structure behaves like a dictionary. It contains all metadata
    that is available for a result. It is inspired by `mne.Info` data structure.
    ResultInfo initialization behaves as a dictionary:
        `result_info = ResultInfo('key1', 'key2', 'length'=40)`
    """

    def __init__(self, source_info, *args, **kwargs):
        super(ResultInfo, self).__init__(**source_info)

        # result info is a dictionary component of the Info class now
        self["result_info"] = dict(*args, **kwargs)

    @property
    def source_fname(self):
        pass

    @property
    def bids_entities(self):
        pass

    @property
    def model_params(self):
        pass

    def save(self, fpath: Union[str, Path]):
        """Save ResultInfo as a JSON dictionary.
        Parameters
        ----------
        fpath : str | pathlib.Path
            The filepath to save the ResultInfo data structure as JSON.
        """
        fname, ext = _parse_ext(fpath, verbose=False)
        if ext != ".json":
            raise RuntimeError(
                "Saving Result Info metadata "
                f"as {ext} is not supported. "
                "Please use .json"
            )

        with open(fpath, "w") as fout:
            json.dump(self, fout,
                      cls=NumpyEncoder,
                      indent=4, sort_keys=True)


class Result(RawArray):
    """
    A Container for EZTrack results, which inherits from MNE.io.RawArray.
    In addition to typical MNE style data structure, this container
    allows a metadata Dictionary to store arbitrary data inside a dictionary.
    Parameters
    ----------
    data : np.ndarray
    info : mne.Info
    metadata : dict
    verbose : bool
    """

    def __init__(
            self,
            data: np.ndarray,
            info: mne.Info,
            metadata: Dict = None,
            verbose: bool = None,
    ):
        resultinfo = ResultInfo(source_info=info, **metadata)
        super(Result, self).__init__(
            data, resultinfo, first_samp=0, copy="auto", verbose=verbose
        )
        # self.metadata = ResultInfo(**metadata)

    def get_metadata(self):
        """Return the metadata Dictionary."""
        return self.info["result_info"]

    def normalize(self, method=None, axis=0):
        """Apply normalization scheme of fragility."""
        if axis == 0:
            channel_wise = False
        elif axis == 1:
            channel_wise = True

        if method is None:
            self.apply_function(
                Normalize.compute_fragilitymetric, channel_wise=channel_wise
            )


def _add_desc_to_bids_fname(bids_fname, description, verbose: bool = True):
    if "desc" in str(bids_fname):
        return bids_fname

    bids_fname, ext = _parse_ext(bids_fname, verbose)

    # split by the datatype
    datatype = bids_fname.split("_")[-1]
    source_bids_fname = bids_fname.split(f"_{datatype}")[0]

    result_fname = source_bids_fname + f"_desc-{description}" + f"_{datatype}" + ext
    return result_fname


def create_bids_layout(
        bids_root: Union[str, Path],
        database_path: Union[str, Path],
        reset_database: bool = False,
) -> bids.BIDSLayout:
    """Generate a pybids layout that is saved locally."""
    layout = bids.layout.BIDSLayout(
        root=bids_root,
        validate=True,
        absolute_paths=True,
        derivatives=False,
        database_path=database_path,
        reset_database=reset_database,
        index_metadata=False,
    )
    return layout
