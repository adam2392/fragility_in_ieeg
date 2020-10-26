import collections
import json
import os
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
from mne_bids.path import get_entities_from_fname
from natsort import natsorted
from rerf.rerfClassifier import rerfClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample

from eztrack.base.publication.study import (
    format_supervised_dataset,
    _sequential_aggregation,
    tune_hyperparameters,
)
from eztrack.io import read_clinical_excel
from eztrack.utils import NumpyEncoder

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams["font.sans-serif"] = "Comic Sans MS"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams["font.family"] = "sans-serif"

# set seed and randomness for downstream reproducibility
seed = 123456
np.random.seed(seed)

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

# BIDS related directories
bids_root = Path("/Volumes/Seagate Portable Drive/data")
bids_root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
bids_root = Path("/home/adam2392/hdd2/Dropbox/epilepsy_bids/")

deriv_path = "/Users/adam2392/Dropbox/epilepsy_bids/derivatives/"
deriv_path = "/home/adam2392/hdd2/Dropbox/epilepsy_bids/derivatives/"

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
# excel_fpath = Path(
#     "/Users/adam2392/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
# )

# to perform the experiment
expname = "sliced"
featuremodels = [
    "fragility",
]

centers = [
    "nih",
    "jhu",
    "ummc",
    "umf",
    "clevelandtvb",
    "clevelandnl",
    "cleveland",
]
normname = "fragility"


def average_roc(fpr, tpr):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 200)

    n_splits = len(fpr)
    for i in range(n_splits):
        interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(mean_fpr, interp_tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    return mean_fpr, tprs, aucs


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
        subjects_map[params["sub"]] = 1

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


def load_data():
    # load in data as patient dictionary of lists
    for i, modelname in enumerate(featuremodels):
        datadir = Path(
            f"/home/adam2392/hdd/derivatives/{expname}/{modelname}"
        )  # "/{reference}/{modality}")
        # datadir = Path(f"/Users/adam2392/Dropbox/{expname}/{modelname}")  # /{reference}/{modality}")

        # load in sliced datasets
        patient_result_dict = _load_patient_dict(datadir, kind=kind, verbose=True)

    print(patient_result_dict.keys())
    patient_aggregation_method = None

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
        #     if not all([_sozinds_list[0] == sozinds for sozinds in _sozinds_list]):
        #         print(f'{subject} has some issues w/ soz inds!')

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


def run_experiment(unformatted_X, y, subjects, sozinds_list, onsetwin_list, clf_type):
    metric = "roc_auc"
    random_state = 12345

    BOOTSTRAP = False
    # define hyperparameters
    windows = [
        (0, 40),
        (-40, 25),  # -5 seconds to first 20% of seizure
        (0, 80),
        (-40, 0),
        (-80, 0),
    ]
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.75]
    weighting_funcs = [None]  # _exponential_weight, _gaussian_weight]

    max_depth = [None, 5, 10]
    max_features = [
        "auto",
        # 'log2'
    ]

    # initialize the classifier
    model_params = {
        "n_estimators": 500,
        "max_depth": max_depth[0],
        "max_features": max_features[0],
        "n_jobs": -1,
        "random_state": random_state,
    }

    if clf_type == "srerf":
        model_params.update(
            {
                "projection_matrix": "S-RerF",
                "image_height": 4,
                "patch_height_max": 4,
                "patch_height_min": 1,
                "patch_width_max": 8,
                "patch_width_min": 1,
            }
        )
        clf_func = rerfClassifier
    elif clf_type == "mtmorf":
        model_params.update(
            {
                "projection_matrix": "MT-MORF",
                "image_height": 4,
                "patch_height_max": 4,
                "patch_height_min": 1,
                "patch_width_max": 8,
                "patch_width_min": 1,
            }
        )
        clf_func = rerfClassifier

    dataset_params = {"sozinds_list": sozinds_list, "onsetwin_list": onsetwin_list}

    # format supervised learning datasets
    # define preprocessing to convert labels/groups into numbers
    enc = OrdinalEncoder()  # handle_unknown='ignore', sparse=False
    subject_groups = enc.fit_transform(np.array(subjects)[:, np.newaxis])
    unformatted_X = np.array(unformatted_X)
    y = np.array(y)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    y_labels = np.array(enc.fit_transform(y))
    subjects = np.array(subjects)

    # re-assign variable names
    groups = subjects
    y = y_labels

    nested_scores = collections.defaultdict(list)

    # create held-out test dataset
    # create separate pool of subjects for testing dataset
    # 1. Cross Validation Training / Testing Split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=random_state)
    for jdx, (train_inds, test_inds) in enumerate(gss.split(unformatted_X, y, groups)):
        # create an iterator of all possible hyperparameters
        hyperparameters = product(windows, thresholds, weighting_funcs)

        # note that training data (Xtrain, ytrain) will get split again
        # X_train, y_train = unformatted_X[train_inds, ...], y[train_inds]
        # groups_train = groups[train_inds]
        #
        # # testing dataset (held out until evaluation)
        # X_test, y_test = unformatted_X[test_inds, ...], y[test_inds]
        # groups_test = groups[test_inds]
        # test_sozinds_list = np.array(sozinds_list)[test_inds]
        # test_onsetwin_list = np.array(onsetwin_list)[test_inds]

        print(unformatted_X.shape, y.shape, groups.shape)
        # tune hyperparameters
        master_scores = tune_hyperparameters(
            clf_func,
            unformatted_X=unformatted_X.copy(),
            y=y,
            groups=groups,
            train_inds=train_inds,
            test_inds=test_inds,
            hyperparameters=hyperparameters,
            dataset_params=dataset_params,
            **model_params,
        )

        # get the best classifier based on pre-chosen metric
        train_key = f"train_{metric}"
        test_key = f"test_{metric}"
        metric_list = [np.mean(scores[test_key]) for scores in master_scores]
        best_index = np.argmax(metric_list)

        # get the best estimator within that inner cv
        best_metric_ind = np.argmax(master_scores[best_index]["test_roc_auc"])
        best_estimator = master_scores[best_index]["estimator"][best_metric_ind]
        best_hyperparameter = master_scores[best_index]["hyperparameters"]
        best_window, best_threshold, _ = best_hyperparameter

        # apply formatting to the dataset
        X_formatted = format_supervised_dataset(
            unformatted_X.copy(),
            **dataset_params,
            window=best_window,
            threshold=best_threshold,
            weighting_func=None,
        )

        # evaluate on the testing dataset
        X_test, y_test = X_formatted[test_inds, ...], y[test_inds]
        groups_test = groups[test_inds]

        # resample the held-out test data via bootstrap
        test_sozinds_list = np.asarray(dataset_params["sozinds_list"])[test_inds]
        test_onsetwin_list = np.asarray(dataset_params["onsetwin_list"])[test_inds]

        if BOOTSTRAP:
            for i in range(500):
                X_boot, y_boot, sozinds, onsetwins = resample(
                    X_test,
                    y_test,
                    test_sozinds_list,
                    test_onsetwin_list,
                    n_samples=len(y_test),
                )
        else:
            X_boot, y_boot = X_test, y_test

        y_pred_prob = best_estimator.predict_proba(X_boot)[:, 1]
        y_pred = best_estimator.predict(X_boot)

        # store analysis done on the validation group
        nested_scores["validate_groups"] = groups_test
        nested_scores["hyperparameters"] = best_hyperparameter
        # pop estimator
        nested_scores["estimator"] = best_estimator

        nested_scores["validate_ytrue"].append(y_test)
        nested_scores["validate_ypred_prob"].append(y_pred_prob)

        # store ROC curve metrics on the held-out test set
        fpr, tpr, thresholds = roc_curve(y_boot, y_pred_prob, pos_label=1)
        fnr, tnr, neg_thresholds = roc_curve(y_boot, y_pred_prob, pos_label=0)
        nested_scores["validate_fpr"].append(fpr)
        nested_scores["validate_tpr"].append(tpr)
        nested_scores["validate_fnr"].append(fnr)
        nested_scores["validate_tnr"].append(tnr)

    return nested_scores


if __name__ == "__main__":
    clf_type = "mtmorf"

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

    print(len(pats_to_avg))

    unformatted_X, y, subjects, sozinds_list, onsetwin_list = load_data()
    nested_scores = run_experiment(
        unformatted_X, y, subjects, sozinds_list, onsetwin_list, clf_type=clf_type
    )

    # summarize the boot-strapped samples
    # fprs, tprs = nested_scores['validate_fpr'], nested_scores['validate_tpr']
    # # aucs = [auc(fpr, tpr) for fpr, tpr in zip(fprs, tprs)]
    # mean_fpr, tprs, aucs = average_roc(fprs, tprs)
    # mean_tpr, std_tpr = np.mean(tprs, axis=0), np.std(tprs, axis=0)
    # # avg/std of the AUC statistic
    # mean_auc = np.mean(aucs)
    # std_auc = np.std(aucs)
    # # plot ROC curve
    # ax = _plot_roc_curve(mean_tpr, mean_fpr, std_tpr=std_tpr, mean_auc=mean_auc, std_auc=std_auc, label='Fragility')
    # print(len(fprs))

    #### SAVE RESULTS
    study_path = Path(deriv_path) / "study"

    # nested CV estimators
    clf_func_path = study_path / "clf" / f"{clf_type}_classifiers_fragility.npz"
    clf_func_path.parent.mkdir(exist_ok=True, parents=True)

    # nested CV scores
    nested_scores_fpath = (
        study_path / f"study_nested_scores_{clf_type}_fragility_6.json"
    )
    estimators = nested_scores.pop("estimator")
    print(nested_scores.keys())

    # save the estimators
    if clf_type not in ["srerf", "mtmorf"]:
        np.savez_compressed(clf_func_path, estimators=estimators)

    # save all the master scores as a JSON file
    with open(nested_scores_fpath, "w") as fin:
        json.dump(nested_scores, fin, cls=NumpyEncoder)
