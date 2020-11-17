"""API for adding bad/good ch_names by setting status."""
from pathlib import Path
from typing import Union

from mne_bids.tsv_handler import _from_tsv, _to_tsv
from mne_bids.path import _find_matching_sidecar
from natsort import natsorted

from eztrack.utils.config import ClinicalColumnns, ClinicalContactColumns, logger
from eztrack.io import read_clinical_excel
from eztrack.preprocess import append_subject_metadata

from eztrack.preprocess.bids_conversion import _update_sidecar_tsv_byname


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

    age = pat_dict.get(ClinicalColumnns.CURRENT_AGE.value, "n/a")
    sex = pat_dict.get(ClinicalColumnns.GENDER.value, "n/a")
    engel_score = pat_dict.get(ClinicalColumnns.ENGEL_SCORE.value, "n/a")
    ilae_score = pat_dict.get(ClinicalColumnns.ILAE_SCORE.value, "n/a")
    cc_score = pat_dict.get(ClinicalColumnns.CLINICAL_COMPLEXITY.value, "n/a")
    handedness = pat_dict.get(ClinicalColumnns.HANDEDNESS.value, "n/a")
    outcome = pat_dict.get(ClinicalColumnns.OUTCOME.value, "n/a")
    date_follow_up = pat_dict.get(ClinicalColumnns.DATE_FOLLOW_UP.value, "n/a")
    ethnicity = pat_dict.get(ClinicalColumnns.ETHNICITY, "n/a")

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

    # add subject level data - Engel score
    subject_metadata = {
        "field": "clinical_complexity",
        "description": "A clinical complexity classification from 1-4. "
                       "All patients are marked in the category that their seizures belong to.",
        "value": engel_score,
        "levels": {
            "1": "lesional",
            "2": "temporal",
            "3": "extratemporal",
            "4": "multifocal",
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

    # add subject level data - ILAE score
    subject_metadata = {
        "field": "ethnicity",
        "description": "The ethnicity of the subject (Caucasian, African American, Hispanic and"
        "Asian are included).",
        "value": ethnicity,
        "levels": {
            "0": "caucasian",
            "1": "african-american",
            "2": "hispanic",
            "3": "asian",
            "-1": "unknown",
        },
    }
    append_subject_metadata(bids_root, subject, **subject_metadata)

    # date of last follow up
    subject_metadata = {
        "field": "date_follow_up",
        "description": "Date of the last follow up, from which outcome, Engel and ILAE score are "
        "derived.",
        "value": date_follow_up,
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

    # subject dir and get all BV files
    subj_dir = Path(Path(bids_root) / f"sub-{subject}")
    fnames = natsorted([x.name for x in subj_dir.rglob(f"*_{kind}.vhdr")])

    # adding bad ch_names to the ch_names tsv
    logger.info(
        f"For {len(fnames)} files, trying to extract bad channels from excel..."
    )
    logger.info(f"Found: {bad_chs}.")

    # adding bad channels to the channels tsv
    for bids_fname in fnames:
        # load in ch_names fpath
        channels_fpath = _find_matching_sidecar(
            bids_fname, bids_root, suffix="channels.tsv"
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
