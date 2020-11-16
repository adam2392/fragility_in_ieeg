"""API for converting files to BIDS format."""
import os
from pathlib import Path
from typing import Union, List

from mne_bids import BIDSPath, read_raw_bids
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from mne_bids.path import get_entities_from_fname, _find_matching_sidecar, BIDSPath
from natsort import natsorted
from tqdm import tqdm

from eztrack.base.utils.file_utils import (
    _get_subject_recordings,
    _get_subject_electrode_layout,
)
from eztrack.io.read_datasheet import read_clinical_excel
from eztrack.preprocess.bids_conversion import (
    write_eztrack_bids,
    append_seeg_layout_info,
    append_subject_metadata,
    append_original_fname_to_scans,
    _bids_validate,
    _update_sidecar_tsv_byname,
)
from eztrack.utils import logger, ClinicalColumnns, ClinicalContactColumns


def add_subject_metadata_from_excel(
    bids_root: Union[Path, str], subject: str, excel_fpath: Union[Path, str]
):
    """Add subject level metadata from excel file."""
    # use excel file to set various data points
    pat_dict = read_clinical_excel(excel_fpath, subject=subject)

    if pat_dict is None:
        return

    if subject.startswith("sub-"):
        subject = subject.split("-")[1]

    age = pat_dict[ClinicalColumnns.CURRENT_AGE.value]
    sex = pat_dict[ClinicalColumnns.GENDER.value]
    engel_score = pat_dict[ClinicalColumnns.ENGEL_SCORE.value]
    ilae_score = pat_dict[ClinicalColumnns.ILAE_SCORE.value]
    handedness = pat_dict[ClinicalColumnns.HANDEDNESS.value]
    outcome = pat_dict[ClinicalColumnns.OUTCOME.value]
    ethnicity = pat_dict[ClinicalColumnns.ETHNICITY.value]
    years_follow_up = pat_dict[ClinicalColumnns.YEARS_FOLLOW_UP.value]
    site = pat_dict[ClinicalColumnns.SITE.value]

    # add subject level data - surgical outcome
    subject_metadata = {
        "field": "outcome",
        "description": "Seizure freedom outcome after surgery.",
        "value": outcome,
        "levels": {
            "S": "successful surgery; seizure freedom",
            "F": "failed surgery; recurring seizures",
            "NR": "no resection/surgery",
        },
    }
    append_subject_metadata(bids_root, subject, **subject_metadata)

    # add subject level data - Engel score
    subject_metadata = {
        "field": "engel_score",
        "description": "A clinical classification from 1-4. See literature for better overview.",
        "value": engel_score,
        "levels": {
            "1": "seizure free",
            "2": "significant seizure frequency reduction",
            "3": "slight seizure frequency reduction",
            "4": "no change",
            "-1": "no surgery, or outcome",
        },
    }
    append_subject_metadata(bids_root, subject, **subject_metadata)

    # add subject level data - ILAE score
    subject_metadata = {
        "field": "ilae_score",
        "description": "A clinical classification from 1-4.",
        "value": ilae_score,
        "levels": {
            "1": "seizure free",
            "2": "",
            "3": "",
            "4": "",
            "5": "",
            "6": "",
            "-1": "no surgery, or outcome",
        },
    }
    append_subject_metadata(bids_root, subject, **subject_metadata)

    # add subject level data - Ethnicity
    subject_metadata = {
        "field": "ethnicity",
        "description": "A coarse description of the ethnicity of subject.",
        "value": ethnicity,
        "levels": {
            "0": "Caucasian",
            "1": "Black",
            "2": "Hispanic",
            "3": "Asian",
        },
    }
    append_subject_metadata(bids_root, subject, **subject_metadata)

    # add subject level data - Years Follow Up
    subject_metadata = {
        "field": "years_follow_up",
        "description": "The number of years of follow up since the surgery.",
        "value": years_follow_up,
    }
    append_subject_metadata(bids_root, subject, **subject_metadata)

    # add subject level data - Years Follow Up
    subject_metadata = {
        "field": "site",
        "description": "The clinical site for the subject",
        "value": site,
        "levels": {
            "nih": "National Institute of Health",
            "ummc": "University of Maryland Medical Center",
            "jhh": "Johns Hopkins Hospital",
            "umf": "University of Miami Florida Hospital",
            "cc": "Cleveland Clinic",
        },
    }
    append_subject_metadata(bids_root, subject, **subject_metadata)

    # get the participants tsv and json
    participants_tsv_fname = Path(Path(bids_root) / "participants.tsv")

    # read participant data
    participants_tsv = _from_tsv(participants_tsv_fname)
    subid = participants_tsv["participant_id"].index(f"sub-{subject}")

    # change age
    participants_tsv["age"][subid] = age
    # change sex
    participants_tsv["sex"][subid] = sex
    # change hand
    participants_tsv["hand"][subid] = handedness

    # write changes to participants tsv file
    _to_tsv(participants_tsv, participants_tsv_fname)


def add_bad_chs_from_excel(
    bids_root: Union[Path, str],
    subject: str,
    excel_fpath: Union[Path, str],
    acquisition: str,
):
    """Append bad ch_names from an excel file."""
    # read in the dataframe of clinical datasheet
    pat_dict = read_clinical_excel(excel_fpath, subject=subject)

    if pat_dict is None:
        return

    # extract the bad channels
    bad_chs = pat_dict[ClinicalContactColumns.BAD_CONTACTS.value]

    # get files of that subject using BIDS Layout
    if acquisition == "eeg":
        kind = "eeg"
    elif acquisition in ["ecog", "seeg"]:
        kind = "ieeg"
    # layout = BIDSLayout(bids_root)
    # bids_filters = {
    #     "subject": subject,
    #     "acquisition": acquisition,
    #     "kind": kind,
    #     "extension": "vhdr",
    # }
    # fnames = layout.get(**bids_filters)
    subj_dir = Path(Path(bids_root) / f"sub-{subject}")
    fnames = natsorted([x.name for x in subj_dir.rglob(f"*_{kind}.vhdr")])

    # adding bad ch_names to the ch_names tsv
    logger.info(
        f"For {len(fnames)} files, trying to extract bad channels from excel..."
    )
    logger.info(f"Found: {bad_chs}.")

    # adding bad channels to the channels tsv
    for bids_fname in fnames:
        # convert to BIDS Path
        if isinstance(bids_fname, str):
            params = get_entities_from_fname(bids_fname)
            bids_fname = BIDSPath(**params)
        # load in ch_names source_fpath
        channels_fpath = _find_matching_sidecar(
            bids_fname, bids_root, kind="channels", extension=".tsv"
        )
        # add `bad` to the `status`
        _update_sidecar_tsv_byname(
            sidecar_fname=channels_fpath,
            name=bad_chs,
            colkey="status",
            val="bad",
            allow_fail=True,
        )
        # add `bad` to the `status`
        _update_sidecar_tsv_byname(
            sidecar_fname=channels_fpath,
            name=bad_chs,
            colkey="status_description",
            val="excel-file-labeled",
            allow_fail=True,
        )


def add_subject_data_from_exceldb(
    bids_root: Union[Path, str], subject_ids: List, excel_fpath: Union[Path, str]
):  # pragma: no cover
    """
    Write subject data from a database, as an Excel file.

    The Excel file should follow BIDs naming convention as closely
    as possible.

    Parameters
    ----------
    bids_root
    subject_ids
    excel_fpath

    """
    # go through each subject
    for subject in subject_ids:
        # append bad channels from excel file
        add_bad_chs_from_excel(
            bids_root,
            subject=subject,
            excel_fpath=excel_fpath,
            acquisition=acquisition,
        )

        # append patient level metadata from excel file
        add_subject_metadata_from_excel(
            bids_root, subject=subject, excel_fpath=excel_fpath
        )
    # validate that this is bids valid dataset
    try:
        _bids_validate(bids_root)
    except Exception as e:
        logger.exception(e)


def _main(
    bids_root, source_path, subject_ids, acquisition, task, session, kind
):  # pragma: no cover
    """Run Bids Conversion script to be updated.

    Just to show example run locally.
    """
    # TODO: this is a hack, so get rid of it eventually
    # set exclusion criterion to find source data
    if task == "ictal":
        exclusion_strs = ["inter", "ii", "aw", "aslp"]
    elif task == "interictal":
        exclusion_strs = ["sz", "seiz", "ictal"]

    # go through each subject
    for subject in subject_ids:
        # TODO: HACK cuz I stored all data as "seeg" instead of
        # subfoldered w/ "ieeg"
        if kind == "eeg":
            subj_dir = Path(source_path / subject / "scalp" / "edf")
        elif kind == "ieeg":
            subj_dir = Path(source_path / subject / "seeg" / "edf")
        # subj_dir = Path(source_path / subject / kind / "edf")

        # HACK: based on patient pools w/o hard-coded acquisitions
        if subject == "nl03" and kind == "ieeg":
            acquisition = "ecog"
        elif kind == "ieeg" and "nl03" in subject_ids:
            acquisition = "seeg"
        if subject == "umf004":
            acquisition = "seeg"
        elif "umf" in subject:
            acquisition = "ecog"
        # acquisition = "ieeg"

        files = _get_subject_recordings(
            subj_dir, subject, exclusion_str=exclusion_strs, ext="edf"
        )
        # pprint(f"In {subj_dir} found {files}")

        # run BIDs conversion for each separate dataset
        for run_id, source_fpath in enumerate(tqdm(natsorted(files)), start=1):
            if any(
                x in source_fpath.as_posix().lower()
                for x in ["tvb18_seeg_sz_2p", "tvb18_seeg_sz_1p"]
            ):
                continue

            logger.info(f"Running run id: {run_id}, with filepath: {source_fpath}")
            bids_basename = BIDSPath(
                subject, session, task, acquisition=acquisition, run=run_id
            )
            bids_fname = bids_basename.copy().update(suffix=f"{kind}.vhdr")
            bids_fpath = (
                Path(bids_root)
                / f"sub-{subject}"
                / f"ses-{session}"
                / kind
                / bids_fname
            )
            if not bids_fpath.exists():
                # convert dataset to BIDS format
                acquisition = bids_basename.acquisition
                if acquisition == "eeg":
                    kind = "eeg"
                elif acquisition in ["ecog", "seeg"]:
                    kind = "ieeg"
                # write to BIDS
                bids_fpath = write_eztrack_bids(source_fpath, bids_basename, bids_root)

                # run validation on the raw data
                raw = read_raw_bids(bids_basename, bids_root)
                raw.load_data()
                raw = raw.drop_channels(raw.info["bads"])
                logger.info(f"Dropping {len(raw.info['bads'])} channels as 'bad'.")
                raw = raw.pick_types(seeg=True, eeg=True, ecog=True)
                # validate_raw_metadata(raw)

            # write other bad channel info
            # if we have
            if acquisition == "seeg":
                electrode_fpath = _get_subject_electrode_layout(
                    source_path,
                    subject,
                )
                # append SEEG layout information
                append_seeg_layout_info(bids_fname, bids_root, electrode_fpath)

            # append scans original filenames
            append_original_fname_to_scans(
                os.path.basename(source_fpath), bids_root, bids_fname
            )

    # validate that this is bids valid dataset
    try:
        _bids_validate(bids_root)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        bids_root = Path("/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_drugs")

        source_dir = bids_root / "sourcedata" / "edf" / "continuous"

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

    # define BIDS identifiers
    acquisition = "eeg"
    task = "drugtaper"
    session = "presurgery"

    # set BIDS kind based on acquistion
    if acquisition in ["ecog", "seeg", "ieeg"]:
        kind = "ieeg"
    elif acquisition in ["eeg"]:
        kind = "eeg"

    fname_map_to_sub = {
        "DA19439": "PY16N011",
        "XA2454": "PY17N014",
        "XA2458": "PY18N013",
    }
    edf_fpaths = [fpath for fpath in source_dir.glob("*.edf") if fpath.is_file()]
    import collections

    output_fpaths = collections.defaultdict(list)
    for fpath in edf_fpaths:
        for fnamekey, subid in fname_map_to_sub.items():
            if fpath.name.startswith(fnamekey):
                output_fpaths[subid].append(fpath)

    for subject, fpaths in output_fpaths.items():
        # run BIDs conversion for each separate dataset
        for run_id, source_fpath in enumerate(tqdm(natsorted(fpaths)), start=1):
            logger.info(f"Running run id: {run_id}, with filepath: {source_fpath}")
            bids_basename = BIDSPath(
                subject, session, task, acquisition=acquisition, run=run_id
            )
            bids_fname = bids_basename.copy().update(suffix=f"{kind}.vhdr")
            bids_fpath = (
                Path(bids_root)
                / f"sub-{subject}"
                / f"ses-{session}"
                / kind
                / bids_fname
            )
            if not bids_fpath.exists():
                # convert dataset to BIDS format
                acquisition = bids_basename.acquisition
                if acquisition == "eeg":
                    kind = "eeg"
                elif acquisition in ["ecog", "seeg"]:
                    kind = "ieeg"
                # write to BIDS
                bids_fpath = write_eztrack_bids(source_fpath, bids_basename, bids_root)

                # run validation on the raw data
                raw = read_raw_bids(bids_basename, bids_root)
                raw.load_data()
                raw = raw.drop_channels(raw.info["bads"])
                logger.info(f"Dropping {len(raw.info['bads'])} channels as 'bad'.")
                raw = raw.pick_types(seeg=True, eeg=True, ecog=True)
                # validate_raw_metadata(raw)

            # append scans original filenames
            append_original_fname_to_scans(
                os.path.basename(source_fpath), bids_root, bids_fname
            )

    # centers = [
    #     # "cleveland",
    #     # "clevelandnl",
    #     # "clevelandtvb",
    #     # "jhu",
    #     # "nih",
    #     # "ummc",
    #     # "umf",
    # ]
    # for center in centers:
    #     # path to original source data
    #     source_path = source_dir / center
    #
    #     # HACK: get all subject ids within sourcedata
    #     subject_ids = natsorted(
    #         [
    #             x.name
    #             for x in source_path.iterdir()
    #             if not x.as_posix().startswith(".")
    #             if x.is_dir()
    #         ]
    #     )
    #     # subject_ids = [
    #     #     "pt1"
    #     #     # "pt17"
    #     #     # "la05"
    #     #     # "umf004"
    #     #     # "pt15"
    #     #     # "la03",
    #     # ]
    #
    #     # run main bids conversion
    #     _main(
    #         bids_root, source_path, subject_ids, acquisition, task, session, kind
    #     )
    #     # add subject metadata
    #     # add_subject_data_from_exceldb(bids_root, subject_ids, excel_fpath)
