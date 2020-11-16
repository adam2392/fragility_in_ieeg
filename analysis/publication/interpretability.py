import collections
import json
import os
from pathlib import Path

import numpy as np
from mne_bids.path import get_entities_from_fname
from natsort import natsorted

from eztrack.base.publication.study import (
    load_patient_dict,
    extract_Xy_pairs,
    _sequential_aggregation,
)
from eztrack.io import read_clinical_excel
from eztrack.utils import NumpyEncoder

# define various list's of patients
separate_pats = [
    "la09",
    "la27",
    "la29",
    "nl02",
    "pt11",
    "tvb7",
    "tvb18",
    "jh107",
]

ignore_pats = [
    "jh107",
]

pats_to_avg = [
    "umf002",
    "umf004",
    "jh103",
    "ummc005",
    "ummc007",
    "ummc008",
    "ummc009",
    "pt8",
    "pt10",
    "pt11",
    "pt12",
    "pt16",
    "pt17",
    "la00",
    "la01",
    "la02",
    "la03",
    "la04",
    "la05",
    "la06",
    "la07",
    "la08",
    "la10",
    "la11",
    "la12",
    "la13",
    "la15",
    "la16",
    "la20",
    "la21",
    "la22",
    "la23",
    "la24",
    "la27",
    "la28",
    "la29",
    "la31",
    "nl01",
    "nl02",
    "nl03",
    "nl04",
    "nl05",
    "nl06",
    "nl07",
    "nl08",
    "nl09",
    "nl13",
    "nl14",
    "nl15",
    "nl16",
    "nl18",
    "nl21",
    "nl23",
    "nl24",
    "tvb1",
    "tvb2",
    "tvb5",
    "tvb7",
    "tvb8",
    "tvb11",
    "tvb12",
    "tvb14",
    "tvb17",
    "tvb18",
    "tvb19",
    "tvb23",
    "tvb27",
    "tvb28",
    "tvb29",
]

# define list of subjects
subjects = [
    "jh101",
    "jh103",
    "jh105",
    "jh108",
    "la00",
    "la01",
    "la02",
    "la03",
    "la04",
    "la05",
    "la06",
    "la07",
    "la08",
    "la09",
    "la10",
    "la11",
    "la12",
    "la13",
    "la15",
    "la16",
    "la17",
    "la20",
    "la21",
    "la22",
    "la23",
    "la24",
    "la27",
    "la28",
    "la29",
    "la31",
    "nl01",
    "nl03",
    "nl04",
    "nl05",
    "nl07",
    "nl08",
    "nl09",
    "nl10",
    "nl13",
    "nl14",
    "nl15",
    "nl16",
    "nl17",
    "nl18",
    "nl19",
    "nl20",
    "nl21",
    "nl22",
    "nl23",
    "nl24",
    "pt1",
    "pt2",
    "pt3",
    "pt6",
    "pt7",
    "pt8",
    "pt10",
    "pt11",
    "pt12",
    "pt13",
    "pt14",
    "pt15",
    "pt16",
    "pt17",
    "tvb1",
    "tvb2",
    "tvb5",
    "tvb7",
    "tvb8",
    "tvb11",
    "tvb12",
    "tvb14",
    "tvb17",
    "tvb18",
    "tvb19",
    "tvb23",
    "tvb27",
    "tvb28",
    "tvb29",
    "umf001",
    "umf002",
    "umf003",
    "umf004",
    "umf005",
    "ummc001",
    "ummc002",
    "ummc003",
    "ummc004",
    "ummc005",
    "ummc006",
    "ummc007",
    "ummc008",
    "ummc009",
]

print(len(pats_to_avg))

# set seed and randomness for downstream reproducibility
seed = 12345
np.random.seed(seed)

# BIDS related directories
bids_root = Path("/Volumes/Seagate Portable Drive/data")
bids_root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
# bids_root = Path("/home/adam2392/hdd2/Dropbox/epilepsy_bids/")

deriv_path = "/Users/adam2392/Dropbox/epilepsy_bids/derivatives/"
# deriv_path = "/home/adam2392/hdd2/Dropbox/epilepsy_bids/derivatives/"

# BIDS entities
session = "presurgery"
acquisition = "seeg"
task = "ictal"
kind = "ieeg"
reference = "average"

# metadata table
excel_fpath = Path(
    "/home/adam2392/hdd2/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
)
excel_fpath = Path(
    "/Users/adam2392/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
)

# to perform the experiment
expname = "sliced"

centers = [
    "nih",
    "jhu",
    "ummc",
    "umf",
    "clevelandtvb",
    "clevelandnl",
    "cleveland",
]

random_state = 12345


def _subsample_matrices_in_time(mat_list):
    maxlen = min([x.shape[1] for x in mat_list])
    if maxlen < 50:
        raise RuntimeError("Preferably not under 50 samples...")

    mat_list = [x[:, :maxlen] for x in mat_list]
    return mat_list


def _load_patient_dict(datadir, kind="ieeg", verbose=True):
    """Load from datadir, sliced datasets as a dictionary <subject>: <list of datasets>."""
    patient_result_dict = collections.defaultdict(list)
    num_datasets = 0

    # get all files inside experiment
    trimmed_npz_fpaths = [x for x in datadir.rglob("*npz")]

    # get a hashmap of all subjects
    subjects_map = {}
    for fpath in trimmed_npz_fpaths:
        params = get_entities_from_fname(
            os.path.basename(fpath).split(f"{expname}-")[1]
        )
        subjects_map[params["subject"]] = 1

    if verbose:
        print(len(subjects_map))

    # loop through each subject
    subject_list = natsorted(subjects_map.keys())
    for subject in subject_list:
        if subject in pats_to_avg:
            #             print("USING AVERAGE for: ", fpath)
            reference = "average"
        else:
            reference = "monopolar"
        subjdir = Path(datadir / reference / kind)
        fpaths = [x for x in subjdir.glob(f"*sub-{subject}_*npz")]

        # load in each subject's data
        for fpath in fpaths:
            # load in the data and append to the patient dictionary data struct
            with np.load(fpath, allow_pickle=True) as data_dict:
                data_dict = data_dict["data_dict"].item()
                patient_result_dict[subject].append(data_dict)

            num_datasets += 1

    if verbose:
        print("Got ", num_datasets, " datasets.")
        print("Got ", len(patient_result_dict), " patients")
        print(patient_result_dict.keys())

    return patient_result_dict


def load_ictal_frag_data(deriv_path, patient_aggregation_method=None):
    modelname = "fragility"

    # load in data as patient dictionary of lists
    datadir = Path(deriv_path) / f"{expname}/{modelname}"  # "/{reference}/{modality}")
    # load in sliced datasets
    patient_result_dict = _load_patient_dict(datadir, kind=kind, verbose=True)

    unformatted_X = []
    y = []
    sozinds_list = []
    onsetwin_list = []
    subjects = []

    for subject, datasets in patient_result_dict.items():
        #     print(subject, len(datasets))
        # read in Excel database
        pat_dict = read_clinical_excel(excel_fpath, subject=subject)
        soz_chs = pat_dict["SOZ_CONTACTS"]
        #     soz_chs = pat_dict['RESECTED_CONTACTS']
        outcome = pat_dict["OUTCOME"]

        # extract list of the data from subject
        mat_list = [dataset["mat"] for dataset in datasets]
        mat_list = _subsample_matrices_in_time(mat_list)
        ch_names_list = [dataset["ch_names"] for dataset in datasets]

        # get a nested list of all the SOZ channel indices
        _sozinds_list = [
            [ind for ind, ch in enumerate(ch_names_list[0]) if ch in soz_chs]
        ] * len(datasets)

        # get a list of all the onset indices in the heatmaps
        _onsetwin_list = [dataset["onsetwin"] for dataset in datasets]
        if subject in ignore_pats:
            continue
        if not _sozinds_list:
            print(subject)
        if all(sozlist == [] for sozlist in _sozinds_list):
            print("Need to skip this subject: ", subject)
            continue

        if outcome == "NR":
            continue

        # aggregate and get (X,Y) pairs
        if patient_aggregation_method == "median":
            if len(np.unique(_onsetwin_list)) > 1:
                print(subject, _onsetwin_list)
                raise RuntimeError("Has more then one onset times... can't aggregate.")
            if not all(set(inds) == set(_sozinds_list[0]) for inds in _sozinds_list):
                print(_sozinds_list)
                raise RuntimeError(
                    "Each dataset has different soz inds... can't aggregate."
                )
            if not all(mat.shape[0] == mat_list[0].shape[0] for mat in mat_list):
                raise RuntimeError("Can't aggregate...")

            for mat in mat_list:
                mat = _smooth_matrix(mat, window_len=8)

            mat = np.max(mat_list, axis=0)
            y.append(outcome)
            unformatted_X.append(mat)
            subjects.append(subject)
            sozinds_list.append(_sozinds_list[0])
            onsetwin_list.append(_onsetwin_list[0])
        elif patient_aggregation_method is None:
            _X, _y, _sozinds_list = _sequential_aggregation(
                mat_list, ch_names_list, _sozinds_list, outcome
            )
            if len(_y) != len(_sozinds_list):
                print(subject, len(_y), len(_sozinds_list))
                continue
            unformatted_X.extend(_X)
            y.extend(_y)
            sozinds_list.extend(_sozinds_list)
            onsetwin_list.extend(_onsetwin_list)
            subjects.extend([subject] * len(_y))

    print(
        len(unformatted_X), len(y), len(sozinds_list), len(subjects), len(onsetwin_list)
    )
    return unformatted_X, y, subjects, sozinds_list, onsetwin_list


def _compute_spatial_contrast(unformatted_X, sozinds_list, onsetwin_list):
    spatial_scores = []
    spatial_soz = []
    spatial_sozc = []
    for X, sozinds, onsetwin in zip(unformatted_X, sozinds_list, onsetwin_list):
        nsozinds = [i for i in range(X.shape[0]) if i not in sozinds]

        # compute average
        # ratio = np.mean(X[sozinds, onsetwin - 80 : onsetwin + 10], axis=0) / (
        #     np.mean(X[sozinds, onsetwin - 80 : onsetwin + 10], axis=0)
        #     + np.mean(X[nsozinds, onsetwin - 80 : onsetwin + 10], axis=0)
        # )
        # ratio = X[:, onsetwin - 80 : onsetwin + 10].flatten()
        # ratio = np.std(X[:, onsetwin - 80 : onsetwin + 10], axis=0)
        X[X < 0.7] = 0.0

        # get the 10th and 90th quantiles of SOZ vs SOZ^C
        soz = np.quantile(X[sozinds, onsetwin - 80 : onsetwin + 10], [0.1, 0.9], axis=0)
        sozc = np.quantile(
            X[nsozinds, onsetwin - 80 : onsetwin + 10], [0.1, 0.9], axis=0
        )
        # ratio = soz / sozc
        # spatial_scores.append(ratio)
        spatial_soz.append(soz)
        spatial_sozc.append(sozc)

    return spatial_soz, spatial_sozc
    # return spatial_scores


def _compute_temporal_stability(unformatted_X, sozinds_list):
    temporal_scores = []
    for X, sozinds in zip(unformatted_X, sozinds_list):
        chs_std = np.std(X[sozinds], axis=1)

        temporal_scores.append(chs_std)
    return temporal_scores


def run_exp(
    feature_name, subjects, intermed_fpath=None,
):
    #### SAVE RESULTS
    study_path = Path(deriv_path) / "study"
    patient_aggregation_method = None

    # load unformatted datasets
    # i.e. datasets without data-hyperparameters applied
    if feature_name == "fragility":
        if not intermed_fpath:
            (
                unformatted_X,
                y,
                subject_groups,
                sozinds_list,
                onsetwin_list,
            ) = load_ictal_frag_data(deriv_path)
        else:
            (
                unformatted_X,
                y,
                subject_groups,
                sozinds_list,
                onsetwin_list,
            ) = load_ictal_frag_data(intermed_fpath)
    else:
        if not intermed_fpath:
            feature_subject_dict = load_patient_dict(
                deriv_path, feature_name, task="ictal", subjects=subjects
            )
            # get the (X, y) tuple pairs
            (
                unformatted_X,
                y,
                sozinds_list,
                onsetwin_list,
                subject_groups,
            ) = extract_Xy_pairs(
                feature_subject_dict,
                excel_fpath=excel_fpath,
                patient_aggregation_method=patient_aggregation_method,
                verbose=False,
            )
        else:
            # get the (X, y) tuple pairs
            feature_fpath = intermed_fpath / f"{feature_name}_unformatted.npz"

            with np.load(feature_fpath, allow_pickle=True) as data_dict:
                unformatted_X, y = data_dict["unformatted_X"], data_dict["y"]
                sozinds_list, onsetwin_list, subject_groups = (
                    data_dict["sozinds_list"],
                    data_dict["onsetwin_list"],
                    data_dict["subject_groups"],
                )

    print(
        len(unformatted_X),
        len(y),
        len(subject_groups),
        len(onsetwin_list),
        len(sozinds_list),
    )

    # only obtain (-30 seconds to -20 seconds of data; 10 seconds of data)
    # new_unformatted_X = []
    # new_subject_groups = []
    #
    # for idx, (X, onsetwin) in enumerate(zip(unformatted_X, onsetwin_list)):
    #     if onsetwin <= 40:
    #         print(f"PROBLEM WITH IDX: {idx}.")
    #         continue
    #
    #     # strip until first 40 samples (5 seconds)
    #     new_unformatted_X.append(X[:, 0:40])
    #     new_subject_groups.append(subject_groups[idx])

    # compute interpretability indices
    # spatial_contrast_scores = _compute_spatial_contrast(
    #     unformatted_X, sozinds_list, onsetwin_list
    # )

    spatial_soz, spatial_sozc = _compute_spatial_contrast(
        unformatted_X, sozinds_list, onsetwin_list
    )

    # unformatted_X = new_unformatted_X
    # subject_groups = new_subject_groups

    # compute temporal stability
    # temporal_stability_scores = _compute_temporal_stability(unformatted_X, sozinds_list)

    print("Done analyzing interpretability...")

    # nested CV scores
    nested_scores_fpath = study_path / f"study_intepretability_{feature_name}.json"
    interp_scores = dict()
    interp_scores["spatial_soz"] = spatial_soz
    interp_scores["spatial_sozc"] = spatial_sozc
    # interp_scores["spatial_contrast"] = spatial_contrast_scores
    # interp_scores["temporal_stability"] = temporal_stability_scores
    interp_scores["subjects"] = subject_groups

    # save all the master scores as a JSON file
    with open(nested_scores_fpath, "w") as fin:
        json.dump(interp_scores, fin, cls=NumpyEncoder)


if __name__ == "__main__":
    feature_names = [
        "fragility",
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
        "highgamma",
        "correlation-degree",
        "correlation-centrality",
        "delta-coherence-centrality",
        "theta-coherence-centrality",
        "alpha-coherence-centrality",
        "beta-coherence-centrality",
        "gamma-coherence-centrality",
        "highgamma-coherence-centrality",
        "delta-coherence-degree",
        "theta-coherence-degree",
        "alpha-coherence-degree",
        "beta-coherence-degree",
        "gamma-coherence-degree",
        "highgamma-coherence-degree",
    ]

    for feature_name in feature_names:
        # feature_name = 'delta'
        intermed_fpath = Path(deriv_path) / "baselinesliced"
        run_exp(
            feature_name, subjects, intermed_fpath=intermed_fpath,
        )
