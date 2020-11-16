from pathlib import Path
import os.path as op
from mne_bids import make_bids_basename


class BIDSPath(dict):
    """Create a partial/full BIDS filename from its component parts.

    BIDS filename prefixes have one or more pieces of metadata in them. They
    must follow a particular order, which is followed by this function. This
    will generate the *prefix* for a BIDS filename that can be used with many
    subsequent files, or you may also give a suffix that will then complete
    the file name.

    Note that all parameters are not applicable to each kind of data. For
    example, electrode location TSV files do not need a task field.

    Parameters
    ----------
    subject : str | None
        The subject ID. Corresponds to "sub".
    session : str | None
        The session for a item. Corresponds to "ses".
    task : str | None
        The task for a item. Corresponds to "task".
    acquisition: str | None
        The acquisition parameters for the item. Corresponds to "acq".
    run : int | None
        The run number for this item. Corresponds to "run".
    processing : str | None
        The processing label for this item. Corresponds to "proc".
    recording : str | None
        The recording name for this item. Corresponds to "recording".
    space : str | None
        The coordinate space for an anatomical file. Corresponds to "space".
    prefix : str | None
        The prefix for the filename to be created. E.g., a path to the folder
        in which you wish to create a file with this name.
    suffix : str | None
        The suffix for the filename to be created. E.g., 'audio.wav'.
    Returns
    -------
    filename : str
        The BIDS filename you wish to create.
    Examples
    --------
    >>> print(make_bids_basename(subject='test', session='two', task='mytask', suffix='data.csv')) # noqa: E501
    sub-test_ses-two_task-mytask_data.csv
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, value):
        if key not in (
            "sub",
            "ses",
            "task",
            "acq",
            "proc",
            "acq",
            "run",
            "rec",
            "space",
            "suffix",
            "data_path",
        ):
            raise ValueError("Key must be one of blah, got %s" % key)
        return dict.__setitem__(self, key, value)

    def _get_name(self):
        keys = (
            "sub",
            "ses",
            "task",
            "acq",
            "proc",
            "acq",
            "run",
            "rec",
            "space",
            "suffix",
            "data_path",
        )
        filename = []
        for key in keys[:-2]:
            if key in self:
                filename.append("%s-%s" % (key, self[key]))
        filename = "_".join(filename)

        if "suffix" in self:
            filename += "_" + self["suffix"]

        if "data_path" in self:
            filename = op.join(self["data_path"], filename)
        return filename

    def as_str(self):
        return self._get_name()

    def __str__(self):
        """Return the string representation of the path, suitable for
        passing to system calls."""
        self._str = self._get_name()
        return self._str

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self._get_name())


if __name__ == "__main__":
    WORKSTATION = "lab"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        bids_root = Path("/Users/adam2392/Downloads/vns_epilepsy/")
        bids_root = Path("/Users/adam2392/Downloads/tngpipeline/")

        source_dir = Path("/Users/adam2392/Dropbox/epilepsy_bids/sourcedata")

        output_dir = Path("/Users/adam2392/Downloads/tngpipeline")

        figures_dir = Path(
            "/Users/adam2392/Dropbox/epilepsy_bids/derivatives/fragility/figures"
        )

        # path to excel layout file - would be changed to the datasheet locally
        excel_fpath = Path(
            "/Users/adam2392/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
        )
    elif WORKSTATION == "lab":
        bids_root = Path("/home/adam2392/hdd2/epilepsy_bids/")
        source_dir = Path("/home/adam2392/hdd2/epilepsy_bids/sourcedata")
        excel_fpath = Path(
            "/home/adam2392/hdd2/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
        )

        # output directory
        output_dir = Path("/home/adam2392/hdd")

        # figures directory
        figures_dir = Path(
            "/home/adam2392/hdd2/epilepsy_bids/derivatives/fragility/figures"
        )

    elif WORKSTATION == "test":
        bids_root = Path("/home/adam2392/Documents/eztrack/data/bids_layout/")

    centers = [
        # 'ummc',
        # "jhu",
        # "nih",
        # "umf",
        "cleveland",
        "clevelandnl",
        "clevelandtvb",
    ]
    # BIDS-identifier variables to change
    # task = "monitor"
    # session = "seizure"
    TASK = "ictal"
    SESSION = "presurgery"
    ACQUISITION = "seeg"
    KIND = "ieeg"
    REFERENCE = "average"
    verbose = True
    overwrite = True

    subjects = {}
    soz_subs = {}

    bids_filename = BIDSPath(sub="la01", ses=SESSION, task=TASK, acq=ACQUISITION)
    print(bids_filename)
    print(Path(bids_filename.as_str()) / "stuff")

    bids_filename.update({"sub": "test"})
    print(bids_filename)
    # bids_filename
    # layout = get_bids_layout(bids_root, database_path=BIDS_LAYOUT_DB_PATH)
    #
    # deriv_path = Path(bids_root) / "derivatives" / "freesurfer"
    # for center in centers:
    #     # center = "cleveland"
    #     sourcedir = Path(bids_root / "sourcedata" / center)
    #
    #     # get all subject ids
    #     subject_ids = layout.get_subjects()
    #     subject_ids = [os.path.basename(x) for x in sourcedir.glob("*") if x.is_dir()]
    #     print(subject_ids)
    #
    #     # subject_ids = [
    #     #     'nl04', 'nl05', 'nl06',
    #     #     # 'nl05', 'nl07', 'nl09', 'nl13', 'nl14', 'nl16', 'nl17', 'nl18', 'nl19', 'nl20', 'nl21', 'nl22', 'nl24'
    #     #     # "tvb1",
    #     #     # "tvb2",
    #     #     # "tvb5",
    #     #     # "tvb11",
    #     #     # "tvb12",
    #     #     # "tvb17",
    #     #     # "tvb18",
    #     #     # "tvb28",
    #     # ]
    #
    #     for subject in subject_ids:
    #         fids_mni = mne.coreg.get_mni_fiducials(
    #             subject, subjects_dir=deriv_path, verbose=verbose
    #         )
    #         print(fids_mni)
    #         break
