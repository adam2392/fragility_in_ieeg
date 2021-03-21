import collections
import json
from itertools import product
from pathlib import Path

import numpy as np
from rerf.rerfClassifier import rerfClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample

# functions related to the feature comparison experiment
from analysis.publication.read_datasheet import read_clinical_excel
from analysis.publication.study import (
    load_patient_dict,
    determine_feature_importances,
    extract_Xy_pairs,
    format_supervised_dataset,
    tune_hyperparameters
)
from analysis.publication.utils import NumpyEncoder

# from rerf.urerf import UnsupervisedRandomForest

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
deriv_path = "/home/adam2392/hdd2/Dropbox/epilepsy_bids/derivatives/"

# BIDS entities
session = "presurgery"
acquisition = "seeg"
task = "ictal"
reference = "average"

# metadata table
excel_fpath = Path(
    "/home/adam2392/hdd2/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
)
# excel_fpath = Path(
#     "/Users/adam2392/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
# )

# to perform the experiment
featuremodels = [
    "fragility",
]

feature_names = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "highgamma",
    "correlation-degree",
    "correlation-centrality",
    "delta-coherence-degree",
    "theta-coherence-degree",
    "alpha-coherence-degree",
    "beta-coherence-degree",
    "gamma-coherence-degree",
    "highgamma-coherence-degree",
    "delta-coherence-centrality",
    "theta-coherence-centrality",
    "alpha-coherence-centrality",
    "beta-coherence-centrality",
    "gamma-coherence-centrality",
    "highgamma-coherence-centrality",
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

random_state = 12345

max_depth = [None, 5, 10]
max_features = ["auto", "log2"]

# initialize the classifier
model_params = {
    "n_estimators": 500,
    "max_depth": max_depth[0],
    "max_features": max_features[0],
    "n_jobs": -1,
    "random_state": random_state,
}


def combine_patient_predictions(
        ytrues, ypred_probs, subject_groups, pat_predictions=None, pat_true=None
):
    if pat_predictions is None or pat_true is None:
        pat_predictions = collections.defaultdict(list)
        pat_true = dict()

    # loop through things
    for ytrue, ypred_proba, subject in zip(ytrues, ypred_probs, subject_groups):
        pat_predictions[subject].append(float(ypred_proba))

        if subject not in pat_true:
            pat_true[subject] = ytrue[0]
        else:
            if pat_true[subject] != ytrue[0]:
                raise RuntimeError("wtf subject should all match...")
    return pat_predictions, pat_true


def sample_cv_clinical_complexity(
        subject_groups, study_path, train_size=0.5, n_splits=10
):
    from sklearn.model_selection import StratifiedShuffleSplit

    # create held-out test dataset
    # create separate pool of subjects for testing dataset
    # 1. Cross Validation Training / Testing Split
    subjects = np.unique(subject_groups)

    clinical_complexity = []
    pat_df = read_clinical_excel(excel_fpath, keep_as_df=True)
    for subj in subjects:
        pat_row = pat_df[pat_df["PATIENT_ID"] == subj.upper()]
        clinical_complexity.append(pat_row["CLINICAL_COMPLEXITY"].values[0])
    clinical_complexity = np.array(clinical_complexity).astype(float)

    gss = StratifiedShuffleSplit(
        n_splits=n_splits, train_size=train_size, random_state=random_state
    )
    for jdx, (train_inds, test_inds) in enumerate(
            gss.split(subjects, clinical_complexity)
    ):
        # if jdx != 7:
        #     continue
        train_pats = subjects[train_inds]
        test_pats = subjects[test_inds]

        print(list(zip(train_pats, clinical_complexity[train_inds])))
        print(len(np.argwhere(clinical_complexity[train_inds] == 1.0)))
        print(len(np.argwhere(clinical_complexity[train_inds] == 2.0)))
        print(len(np.argwhere(clinical_complexity[train_inds] == 3.0)))
        print(len(np.argwhere(clinical_complexity[train_inds] == 4.0)))
        np.savez_compressed(
            study_path / "inds" / "clinical_complexity" / f"{jdx}-inds.npz",
            # train_inds=train_inds, test_inds=test_inds,
            train_pats=train_pats,
            test_pats=test_pats,
        )
    return


def run_traintest_exp(
        intermed_fpath=None,
        clf_type="rf",
):
    """Run train-test experiment (no nested cv)."""
    from analysis.publication.extract_datasets import load_ictal_frag_data

    feature_name = "fragility"
    metric = "roc_auc"
    BOOTSTRAP = False

    # initialize the classifier
    model_params = {
        "n_estimators": 500,
        "max_depth": max_depth[0],
        "max_features": max_features[0],
        "n_jobs": -1,
        "random_state": random_state,
    }
    #### SAVE RESULTS
    study_path = Path(deriv_path) / "study"

    # define hyperparameters
    windows = [
        (-80, 25),
        # (0, 40),
        # (-40, 25),  # -5 seconds to first 20% of seizure
        # (0, 80),
        # (-40, 0),
        # (-80, 0),
    ]
    thresholds = [
        None,
    ]
    weighting_funcs = [None]  # _exponential_weight, _gaussian_weight]

    # initialize the type of classification function we'll use
    IMAGE_HEIGHT = 20
    if clf_type == "rf":
        clf_func = RandomForestClassifier
    elif clf_type == "srerf":
        model_params.update(
            {
                "projection_matrix": "S-RerF",
                "image_height": IMAGE_HEIGHT,
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
                "image_height": IMAGE_HEIGHT,
                "patch_height_max": 4,
                "patch_height_min": 1,
                "patch_width_max": 8,
                "patch_width_min": 1,
            }
        )
        clf_func = rerfClassifier

    # load unformatted datasets
    # i.e. datasets without data-hyperparameters applied
    if not intermed_fpath:
        (
            unformatted_X,
            y,
            subject_groups,
            sozinds_list,
            onsetwin_list,
        ) = load_ictal_frag_data(deriv_path, excel_fpath=excel_fpath)
    else:
        (
            unformatted_X,
            y,
            subject_groups,
            sozinds_list,
            onsetwin_list,
        ) = load_ictal_frag_data(intermed_fpath, excel_fpath=excel_fpath)

    print(
        len(unformatted_X),
        len(y),
        len(subject_groups),
        len(onsetwin_list),
        len(sozinds_list),
    )

    # get the dataset parameters loaded in
    dataset_params = {"sozinds_list": sozinds_list, "onsetwin_list": onsetwin_list}

    # format supervised learning datasets
    # define preprocessing to convert labels/groups into numbers
    enc = OrdinalEncoder()  # handle_unknown='ignore', sparse=False
    y = enc.fit_transform(np.array(y)[:, np.newaxis])
    subject_groups = np.array(subject_groups)
    # store the cross validation nested scores per feature
    cv_scores = collections.defaultdict(list)

    # run this without the above for a warm start
    for jdx in range(0, 10):
        cv_scores = collections.defaultdict(list)

        with np.load(
                # study_path / "inds" / 'clinical_complexity' / f"{jdx}-inds.npz",
                study_path
                / "inds"
                / "fixed_folds_subjects"
                / f"fragility-srerf-{jdx}-inds.npz",
                allow_pickle=True,
        ) as data_dict:
            # train_inds, test_inds = data_dict["train_inds"], data_dict["test_inds"]
            train_pats, test_pats = data_dict["train_pats"], data_dict["test_pats"]

        # set train indices based on which subjects
        train_inds = [
            idx for idx, sub in enumerate(subject_groups) if sub in train_pats
        ]
        test_inds = [idx for idx, sub in enumerate(subject_groups) if sub in test_pats]

        # note that training data (Xtrain, ytrain) will get split again
        # testing dataset (held out until evaluation)
        subjects_test = subject_groups[test_inds]
        print(subjects_test)

        if len(np.unique(y[test_inds])) == 1:
            print(f"Skipping group cv iteration {jdx} due to degenerate test set")
            continue

        '''Run cross-validation.'''
        window = windows[0]
        threshold = thresholds[0]
        weighting_func = weighting_funcs[0]
        X_formatted, dropped_inds = format_supervised_dataset(
            unformatted_X,
            **dataset_params,
            window=window,
            threshold=threshold,
            weighting_func=weighting_func,
        )

        # run cross-validation
        # instantiate model
        if clf_func == RandomForestClassifier:
            # instantiate the classifier
            clf = clf_func(**model_params)
        elif clf_func == rerfClassifier:
            model_params.update({"image_width": np.abs(window).sum()})
            clf = clf_func(**model_params)
        else:
            clf = clf_func

        # perform CV using Sklearn
        scoring_funcs = {
            "roc_auc": roc_auc_score,
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "average_precision": average_precision_score,
        }
        scores = cross_validate(
            clf,
            X_formatted,
            y,
            groups=subject_groups,
            cv=[train_inds, test_inds],
            scoring=list(scoring_funcs.keys()),
            return_estimator=True,
            return_train_score=True,
        )

        # get the best classifier based on pre-chosen metric
        test_key = f"test_{metric}"
        print(scores.keys())
        print(scores)
        estimator = scores.pop('estimator')

        # resample the held-out test data via bootstrap
        test_sozinds_list = np.asarray(dataset_params["sozinds_list"])[test_inds]
        test_onsetwin_list = np.asarray(dataset_params["onsetwin_list"])[test_inds]
        # evaluate on the testing dataset
        X_test, y_test = np.array(X_formatted)[test_inds, ...], np.array(y)[test_inds]
        groups_test = np.array(subject_groups)[test_inds]

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
            X_boot, y_boot = X_test.copy(), y_test.copy()

        # evaluate on the test set
        y_pred_prob = estimator.predict_proba(X_boot)[:, 1]
        y_pred = estimator.predict(X_boot)

        # store the actual outcomes and the predicted probabilities
        cv_scores["validate_ytrue"].append(list(y_test))
        cv_scores["validate_ypred_prob"].append(list(y_pred_prob))
        cv_scores["validate_ypred"].append(list(y_pred))
        cv_scores['validate_subject_groups'].append(list(groups_test))

        # store ROC curve metrics on the held-out test set
        fpr, tpr, thresholds = roc_curve(y_boot, y_pred_prob, pos_label=1)
        fnr, tnr, neg_thresholds = roc_curve(y_boot, y_pred_prob, pos_label=0)
        cv_scores["validate_fpr"].append(list(fpr))
        cv_scores["validate_tpr"].append(list(tpr))
        cv_scores["validate_fnr"].append(list(fnr))
        cv_scores["validate_tnr"].append(list(tnr))
        cv_scores["validate_thresholds"].append(list(thresholds))

        print("Done analyzing ROC stats...")

        # run the feature importances
        # compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_boot, y_pred_prob, n_bins=10, strategy="quantile"
        )
        clf_brier_score = np.round(
            brier_score_loss(y_boot, y_pred_prob, pos_label=np.array(y_boot).max()), 2
        )

        print("Done analyzing calibration stats...")

        # store ingredients for a calibration curve
        cv_scores["validate_brier_score"].append(float(clf_brier_score))
        cv_scores["validate_fraction_pos"].append(list(fraction_of_positives))
        cv_scores["validate_mean_pred_value"].append(list(mean_predicted_value))

        # store outputs to run McNemars test and Cochrans Q test
        # get the shape of a single feature "vector" / structure array
        pat_predictions, pat_true = combine_patient_predictions(
            y_boot, y_pred_prob, subjects_test
        )
        cv_scores["validate_pat_predictions"].append(pat_predictions)
        cv_scores["validate_pat_true"].append(pat_true)

        # store output for feature importances
        if clf_type == "rf":
            n_jobs = -1
        else:
            n_jobs = 1
        results = determine_feature_importances(
            estimator, X_boot, y_boot, n_jobs=n_jobs
        )
        imp_std = results.importances_std
        imp_vals = results.importances_mean
        cv_scores["validate_imp_mean"].append(list(imp_vals))
        cv_scores["validate_imp_std"].append(list(imp_std))

        print("Done analyzing feature importances...")

        # save intermediate analyses
        clf_func_path = (
                study_path / "clf-train-vs-test" / f"{clf_type}_classifiers_{feature_name}_{jdx}.npz"
        )
        clf_func_path.parent.mkdir(exist_ok=True, parents=True)

        # nested CV scores
        nested_scores_fpath = (
                study_path / f"study_cv_scores_{clf_type}_{feature_name}_{jdx}.json"
        )

        # save the estimators
        if clf_type not in ["srerf", "mtmorf"]:
            estimators = scores.pop("estimator")
            np.savez_compressed(clf_func_path, estimators=estimators)

        # save all the master scores as a JSON file
        with open(nested_scores_fpath, "w") as fin:
            json.dump(cv_scores, fin, cls=NumpyEncoder)


def run_exp(
        feature_name,
        subjects,
        intermed_fpath=None,
        patient_aggregation_method=None,
        clf_type="rf",
):
    from analysis.publication.extract_datasets import load_ictal_frag_data

    metric = "roc_auc"
    BOOTSTRAP = False
    # initialize the classifier
    model_params = {
        "n_estimators": 500,
        "max_depth": max_depth[0],
        "max_features": max_features[0],
        "n_jobs": -1,
        "random_state": random_state,
    }
    #### SAVE RESULTS
    study_path = Path(deriv_path) / "study"

    # define hyperparameters
    windows = [
        (-80, 25),
        # (0, 40),
        # (-40, 25),  # -5 seconds to first 20% of seizure
        # (0, 80),
        # (-40, 0),
        # (-80, 0),
    ]
    thresholds = [
        # None,
        # 0.1, 0.2, 0.3, 0.4,
        0.5,
        0.6,
        0.7,
        # 0.8, 0.9,
        # 0.4, 0.5, 0.6, 0.7,
        #           0.75
    ]
    weighting_funcs = [None]  # _exponential_weight, _gaussian_weight]

    # initialize the type of classification function we'll use
    IMAGE_HEIGHT = 20
    if clf_type == "rf":
        clf_func = RandomForestClassifier
    elif clf_type == "srerf":
        model_params.update(
            {
                "projection_matrix": "S-RerF",
                "image_height": IMAGE_HEIGHT,
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
                "image_height": IMAGE_HEIGHT,
                "patch_height_max": 4,
                "patch_height_min": 1,
                "patch_width_max": 8,
                "patch_width_min": 1,
            }
        )
        clf_func = rerfClassifier

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
            ) = load_ictal_frag_data(deriv_path, excel_fpath=excel_fpath)
        else:
            (
                unformatted_X,
                y,
                subject_groups,
                sozinds_list,
                onsetwin_list,
            ) = load_ictal_frag_data(intermed_fpath, excel_fpath=excel_fpath)
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

    # get the dataset parameters loaded in
    dataset_params = {"sozinds_list": sozinds_list, "onsetwin_list": onsetwin_list}

    # format supervised learning datasets
    # define preprocessing to convert labels/groups into numbers
    enc = OrdinalEncoder()  # handle_unknown='ignore', sparse=False
    #     subject_groups = enc.fit_transform(np.array(subjects)[:, np.newaxis])
    y = enc.fit_transform(np.array(y)[:, np.newaxis])
    subject_groups = np.array(subject_groups)
    # store the cross validation nested scores per feature
    nested_scores = collections.defaultdict(list)

    print(feature_name)

    estimators = []
    # create held-out test dataset
    # create separate pool of subjects for testing dataset
    # 1. Cross Validation Training / Testing Split
    # gss = GroupShuffleSplit(n_splits=10, train_size=.5, random_state=random_state)
    # for jdx, (train_inds, test_inds) in enumerate(gss.split(unformatted_X, y, subject_groups)):
    #     # if jdx != 7:
    #     #     continue
    #     train_pats = np.unique(subject_groups[train_inds])
    #     test_pats = np.unique(subject_groups[test_inds])
    #     np.savez_compressed(study_path / 'inds' / f'{feature_name}-srerf-{jdx}-inds.npz',
    #                         train_inds=train_inds, test_inds=test_inds,
    #                         train_pats=train_pats, test_pats=test_pats)
    # return

    # utilize the sampling of clinical complexity
    # sample_cv_clinical_complexity(subject_groups, study_path, train_size=.7, n_splits=5)
    # return

    # run this without the above for a warm start
    for jdx in range(6, 7):
        with np.load(
                # study_path / "inds" / 'clinical_complexity' / f"{jdx}-inds.npz",
                study_path
                / "inds"
                / "fixed_folds_subjects"
                / f"fragility-srerf-{jdx}-inds.npz",
                allow_pickle=True,
        ) as data_dict:
            # train_inds, test_inds = data_dict["train_inds"], data_dict["test_inds"]
            train_pats, test_pats = data_dict["train_pats"], data_dict["test_pats"]

        # set train indices based on which subjects
        train_inds = [
            idx for idx, sub in enumerate(subject_groups) if sub in train_pats
        ]
        test_inds = [idx for idx, sub in enumerate(subject_groups) if sub in test_pats]

        # note that training data (Xtrain, ytrain) will get split again
        # testing dataset (held out until evaluation)
        subjects_test = subject_groups[test_inds]
        print(subjects_test)

        if len(np.unique(y[test_inds])) == 1:
            print(f"Skipping group cv iteration {jdx} due to degenerate test set")
            continue

        tune_dataset_hyperparams = True
        if tune_dataset_hyperparams:
            # tune hyperparameters
            try:
                # create an iterator of all possible hyperparameters
                hyperparameters = product(windows, thresholds, weighting_funcs)

                master_scores = tune_hyperparameters(
                    clf_func,
                    unformatted_X=unformatted_X.copy(),
                    y=y.copy(),
                    groups=subject_groups.copy(),
                    train_inds=train_inds.copy(),
                    test_inds=test_inds.copy(),
                    hyperparameters=hyperparameters,
                    dataset_params=dataset_params,
                    **model_params,
                )
            except ValueError as e:
                print(jdx, e)
                continue

            print("Done tuning data hyperparameters...")
        else:
            window = windows[0]
            threshold = thresholds[0]
            weighting_func = weighting_funcs[0]
            X_formatted, dropped_inds = format_supervised_dataset(
                unformatted_X,
                **dataset_params,
                window=window,
                threshold=threshold,
                weighting_func=weighting_func,
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
        X_formatted, dropped_inds = format_supervised_dataset(
            unformatted_X.copy(),
            **dataset_params,
            window=best_window,
            threshold=best_threshold,
            weighting_func=None,
        )

        # evaluate on the testing dataset
        X_test, y_test = np.array(X_formatted)[test_inds, ...], np.array(y)[test_inds]
        groups_test = np.array(subject_groups)[test_inds]

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
            X_boot, y_boot = X_test.copy(), y_test.copy()

        y_pred_prob = best_estimator.predict_proba(X_boot)[:, 1]
        y_pred = best_estimator.predict(X_boot)

        # store analysis done on the validation group
        nested_scores["validate_groups"].append(groups_test)
        nested_scores["validate_subjects"].append(subjects_test)
        nested_scores["hyperparameters"].append(best_hyperparameter)

        if clf_type == "rf":
            # pop estimator
            nested_scores["estimator"].append(best_estimator)
        estimators.append(best_estimator)

        # store the actual outcomes and the predicted probabilities
        nested_scores["validate_ytrue"].append(list(y_test))
        nested_scores["validate_ypred_prob"].append(list(y_pred_prob))
        nested_scores["validate_ypred"].append(list(y_pred))

        # store ROC curve metrics on the held-out test set
        fpr, tpr, thresholds = roc_curve(y_boot, y_pred_prob, pos_label=1)
        fnr, tnr, neg_thresholds = roc_curve(y_boot, y_pred_prob, pos_label=0)
        nested_scores["validate_fpr"].append(list(fpr))
        nested_scores["validate_tpr"].append(list(tpr))
        nested_scores["validate_fnr"].append(list(fnr))
        nested_scores["validate_tnr"].append(list(tnr))
        nested_scores["validate_thresholds"].append(list(thresholds))

        print("Done analyzing ROC stats...")

        # run the feature importances
        # compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_boot, y_pred_prob, n_bins=10, strategy="quantile"
        )
        clf_brier_score = np.round(
            brier_score_loss(y_boot, y_pred_prob, pos_label=np.array(y_boot).max()), 2
        )

        print("Done analyzing calibration stats...")

        # store ingredients for a calibration curve
        nested_scores["validate_brier_score"].append(float(clf_brier_score))
        nested_scores["validate_fraction_pos"].append(list(fraction_of_positives))
        nested_scores["validate_mean_pred_value"].append(list(mean_predicted_value))

        # store outputs to run McNemars test and Cochrans Q test
        # get the shape of a single feature "vector" / structure array
        pat_predictions, pat_true = combine_patient_predictions(
            y_boot, y_pred_prob, subjects_test
        )
        nested_scores["validate_pat_predictions"].append(pat_predictions)
        nested_scores["validate_pat_true"].append(pat_true)

        # store output for feature importances
        X_shape = X_boot[0].reshape((IMAGE_HEIGHT, -1)).shape
        if clf_type == "rf":
            n_jobs = -1
        else:
            n_jobs = 1
        results = determine_feature_importances(
            best_estimator, X_boot, y_boot, n_jobs=n_jobs
        )
        imp_std = results.importances_std
        imp_vals = results.importances_mean
        nested_scores["validate_imp_mean"].append(list(imp_vals))
        nested_scores["validate_imp_std"].append(list(imp_std))

        print("Done analyzing feature importances...")

        # save intermediate analyses
        clf_func_path = (
                study_path / "clf" / f"{clf_type}_classifiers_{feature_name}_{jdx}.npz"
        )
        clf_func_path.parent.mkdir(exist_ok=True, parents=True)

        # nested CV scores
        nested_scores_fpath = (
                study_path / f"study_nested_scores_{clf_type}_{feature_name}_{jdx}.json"
        )

        # save the estimators
        if clf_type not in ["srerf", "mtmorf"]:
            estimators = nested_scores.pop("estimator")
            np.savez_compressed(clf_func_path, estimators=estimators)

        # save all the master scores as a JSON file
        with open(nested_scores_fpath, "w") as fin:
            json.dump(nested_scores, fin,
                      cls=NumpyEncoder
                      )

        del master_scores
        del estimators
        del best_estimator

    # nested CV estimators
    # clf_func_path = study_path / 'clf' / f'{clf_type}_classifiers_{feature_name}.npz'
    # clf_func_path.parent.mkdir(exist_ok=True, parents=True)
    #
    # # nested CV scores
    # nested_scores_fpath = study_path / f'study_nested_scores_{clf_type}_{feature_name}.json'
    #
    # # save the estimators
    # if clf_type not in ['srerf', 'mtmorf']:
    #     estimators = nested_scores.pop('estimator')
    #     np.savez_compressed(clf_func_path, estimators=estimators)
    #
    # # save all the master scores as a JSON file
    # with open(nested_scores_fpath, 'w') as fin:
    #     json.dump(nested_scores, fin, cls=NumpyEncoder)


if __name__ == "__main__":
    expname = "sliced"
    excel_fpath = Path(
        "/home/adam2392/hdd2/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
    )
    feature_names = [
        # "fragility",
        # "delta",
        # "theta",
        # "alpha",
        # "beta",
        # "gamma",
        # "highgamma",
        # "correlation-degree",
        # "correlation-centrality",
        # 'delta-coherence-centrality', 'theta-coherence-centrality', 'alpha-coherence-centrality',
        # 'beta-coherence-centrality', 'gamma-coherence-centrality', 'highgamma-coherence-centrality',
        # "delta-coherence-degree",
        # 'theta-coherence-degree', 'alpha-coherence-degree',
        # 'beta-coherence-degree', 'gamma-coherence-degree',
        "highgamma-coherence-degree",
    ]

    train_size = 0.6

    for feature_name in feature_names:
        # feature_name = 'delta'
        intermed_fpath = Path(deriv_path) / "baselinesliced"
        # intermed_fpath = Path(f"/Users/adam2392/Dropbox/")
        clf_type = "mtmorf"
        run_exp(
            feature_name,
            subjects,
            intermed_fpath=intermed_fpath,
            patient_aggregation_method=None,
            clf_type=clf_type,
        )
