import json
import os
from pathlib import Path
import shutil

import numpy as np
from mne_bids import make_bids_basename
from mne_bids.path import get_entities_from_fname


def _convert_fpath(fpath, session=None, task=None, copy=True, verbose=True):
    basedir = os.path.dirname(fpath)
    fname = os.path.basename(fpath)
    params = get_entities_from_fname(fname, verbose=verbose)

    if session is not None:
        fname = fname.replace(f"ses-{params['ses']}", f"ses-{session}")
    if task is not None:
        fname = fname.replace(f"task-{params['task']}", f"task-{task}")

    new_fpath = os.path.join(basedir, fname)
    if copy:
        os.rename(fpath, new_fpath)
        # shutil.copyfile(fpath, new_fpath)
    return new_fpath


def _update_metadata(json_fpath, session, task):
    with open(json_fpath, "r") as fin:
        metadata = json.load(fin)

    # get existing seizure event markers
    sz_onset = metadata["onsetwin"]
    clin_onset = metadata["clinonset_win"]
    sz_offset = metadata["offsetwin"]

    events, events_id = np.array(metadata["events"]), np.array(metadata["events_id"])

    if sz_onset is None:
        sz_onset_id = None
    else:
        # ind = events[:, 0].index(sz_onset)
        ind = np.where(events[:, 0] == sz_onset)[0][0]
        sz_onset_id = int(events[ind, 2])

    # seizure offset might be set to the very end of the snapshot if
    # label annotation is not available
    if sz_offset is None or sz_offset == metadata["numwins"]:
        sz_offset_id = sz_offset
    else:
        ind = np.where(events[:, 0] == sz_offset)[0][0]
        sz_offset_id = int(events[ind, 2])
    if clin_onset is None:
        clin_onset_id = clin_onset
    else:
        ind = np.where(events[:, 0] == clin_onset)[0][0]
        clin_onset_id = int(events[ind, 2])

    # onsets = events[:, 0] / raw.info['sfreq']
    # durations = np.zeros_like(onsets)  # assumes instantaneous events
    # descriptions = [mapping[event_id] for event_id in events[:, 2]]
    # annot_from_events = mne.Annotations(onset=onsets, duration=durations,
    #                                     description=descriptions,
    #                                     orig_time=raw.info['meas_date'])
    # update seizure event markers
    metadata["sz_onset_event_id"] = sz_onset_id
    metadata["clin_onset_event_id"] = clin_onset_id
    metadata["sz_offset_event_id"] = sz_offset_id

    # update output filename
    basedir = os.path.dirname(json_fpath)
    fname = os.path.basename(json_fpath)
    params = get_entities_from_fname(fname)
    if session is not None:
        fname = fname.replace(f"ses-{params['session']}", f"ses-{session}")
    if task is not None:
        fname = fname.replace(f"task-{params['task']}", f"task-{task}")
    output_fpath = os.path.join(basedir, fname)

    metadata["output_fname"] = fname.replace("json", "npz")

    # save resulting dictionary back into json
    with open(output_fpath, "w") as fout:
        json.dump(metadata, fout, indent=4, sort_keys=True)
    return metadata


def convert_dataset_derivatives(
    deriv_dir, session="presurgery", task="ictal", subjects=None, verbose=True
):
    if subjects is None:
        subjects = [
            x.name
            for x in Path(deriv_dir).glob("*")
            if not x.as_posix().startswith(".")
        ]

    for subject in subjects:
        print("Converting ", subject)
        subject_dir = Path(deriv_dir) / subject
        fpaths = [
            x
            for x in subject_dir.rglob("*")
            if subject in x.name
            if not x.name.startswith(".")
        ]

        for fpath in fpaths:
            params = get_entities_from_fname(os.path.basename(fpath))
            if params["session"] == session or params["task"] == task:
                continue
            print("Converting ", fpath)
            if fpath.suffix == ".npz":
                _convert_fpath(fpath, session=session, task=task, verbose=verbose)

            if fpath.suffix == ".json":
                _update_metadata(
                    fpath,
                    session,
                    task,
                )
                print("Deleting ", fpath)
                os.remove(fpath)

        # for fpath in fpaths:
        #     print("Deleting ", fpath)
        #     os.remove(fpath)

        print("Done!")


def convert_dataset_entities(
    bids_root, src_bids_entities, dest_bids_entities, subject=None
):
    """
    Convert datasets within a bids_root from a set of BIDS entities into another.

    May convert either all subjects inside 'bids_root', or just one subject, depending
    on parameter 'subject'.

    Parameters
    ----------
    bids_root : str
    src_bids_entities : dict
    dest_bids_entities : dict
    subject : str
        (Optional) Only converts datasets for this subject if passed in. Else, defaults
        to None, which will convert all datasets within bids_root.
    Returns
    -------

    """
    # convert to Posix
    bids_root = Path(bids_root)
    if subject is None:
        subjects = [sub.name for sub in bids_root.glob("sub-*") if sub.is_dir()]
    else:
        subjects = [f"sub-{subject}"]

    bids_entities = [
        "session",
        "task",
        "acquisition",
    ]
    if any([entity not in bids_entities for entity in src_bids_entities]):
        raise RuntimeError(f"Source entities allowed are {bids_entities}.")
    if any([entity not in bids_entities for entity in dest_bids_entities]):
        raise RuntimeError(f"Destination entities allowed are {bids_entities}.")

    for idx, subject in enumerate(subjects):
        subj_dir = Path(bids_root / subject)

        for entity in bids_entities:
            pre_entity = src_bids_entities.get(entity, None)
            dest_entity = dest_bids_entities.get(entity, None)

            if pre_entity is not None and dest_entity is not None:
                if entity == "session":
                    session_dir = Path(subj_dir / f"ses-{pre_entity}")
                    fpaths = [
                        x
                        for x in session_dir.rglob("*")
                        if subject in x.name
                        if pre_entity in x.name
                    ]
                    for fpath in fpaths:
                        pass
                    # rename files within scans
                    # src_scans_fname = make_bids_basename(subject, session=pre_entity,
                    #                          suffix='scans.tsv', prefix=src)
                    # scans_tsv = _from_tsv(src_scans_fname)
                    # for jdx, fname in enumerate(scans_tsv['filename']):

                    # rename scans_fname
                    # src = s/rc_scans_fname
                    dst = make_bids_basename(
                        subject, session=dest_entity, suffix="scans.tsv", prefix=dst
                    )
                    os.rename(src, dst)

                    # rename folder
                    src = Path(subj_dir / f"ses-{pre_entity}")
                    dst = Path(subj_dir / f"ses-{dest_entity}")
                    if os.path.exists(src):
                        os.rename(src, dst)
                    else:
                        raise RuntimeError(f"Source folder/file does not exist: {src}")

                elif entity == "task":
                    pass
                elif entity == "acquisition":
                    pass

    return bids_root


if __name__ == "__main__":
    WORKSTATION = "lab"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        bids_root = Path("/Users/adam2392/Downloads/vns_epilepsy/")
        bids_root = Path("/Users/adam2392/Downloads/tngpipeline/")

        source_dir = Path("/Users/adam2392/Dropbox/epilepsy_bids/sourcedata")

        deriv_dir = Path(
            "/Users/adam2392/Downloads/tngpipeline/derivatives/fragility/monopolar/"
        )

        figures_dir = Path(
            "/Users/adam2392/Dropbox/epilepsy_bids/derivatives/fragility/figures"
        )

        # path to excel layout file - would be changed to the datasheet locally
        excel_fpath = Path(
            "/Users/adam2392/Dropbox/epilepsy_bids/organized_clinical_datasheet_raw.xlsx"
        )
    elif WORKSTATION == "lab":
        bids_root = Path("/home/adam2392/hdd2/epilepsy_bids/")
        source_dir = Path("/home/adam2392/hdd2/epilepsy_bids/sourcedata")
        # output directory
        deriv_dir = Path("/home/adam2392/hdd/derivatives/fragility/average")

        # figures directory
        figures_dir = Path(
            "/home/adam2392/hdd2/epilepsy_bids/derivatives/fragility/figures"
        )

    elif WORKSTATION == "test":
        bids_root = Path("/home/adam2392/Documents/eztrack/data/bids_layout/")

    convert_dataset_derivatives(deriv_dir)
