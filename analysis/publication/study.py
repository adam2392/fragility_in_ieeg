import collections
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
from joblib import cpu_count
from rerf.rerfClassifier import rerfClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
)
import scipy
import scipy.signal
from sklearn.model_selection import GroupKFold, cross_validate

from analysis.publication.data_structure import Result
from analysis.publication.read_datasheet import read_clinical_excel
from analysis.publication.utils import (Normalize, _select_window,
                                        _get_onset_event_id, _resample_mat,
                                        _smooth_matrix, _apply_threshold, _resample_seizure)

# make jobs use half the CPU count
num_cores = cpu_count() // 2


def extract_Xy_pairs(
        patient_result_dict, excel_fpath, patient_aggregation_method=None, verbose=True
):
    """Only works for non-fragility sliced data..."""
    X = []
    y = []
    sozinds_list = []
    onsetwin_list = []
    subjects = []

    for subject, datasets in patient_result_dict.items():
        # read in Excel database
        pat_dict = read_clinical_excel(excel_fpath, subject=subject)
        soz_chs = pat_dict["SOZ_CONTACTS"]
        #     soz_chs = pat_dict['RESECTED_CONTACTS']
        outcome = pat_dict["OUTCOME"]

        print(subject, len(datasets))
        # get a nested list of all the SOZ channel indices
        _sozinds_list = [
                            [ind for ind, ch in enumerate(datasets[0].ch_names) if ch in soz_chs]
                        ] * len(datasets)

        for idx, result in enumerate(datasets):
            if max(_sozinds_list[0]) > len(result.ch_names):
                print(f"Deleting one result for {subject}")
                del datasets[idx]
                # mat_list = np.delete(mat_list, idx, axis=0)
                _sozinds_list = _sozinds_list[:-1]

        # extract list of the data from subject
        mat_list = [result.get_data() for result in datasets]
        ch_names_list = [result.ch_names for result in datasets]

        if verbose:
            print(subject)
            print(len(ch_names_list))
            # print(datasets)

        # get a list of all the onset indices in the heatmaps
        _onsetwin_list = [result.get_metadata()["sz_onset_win"] for result in datasets]
        if not _sozinds_list:
            print(subject)
        if all([sozlist == [] for sozlist in _sozinds_list]):
            print("Need to skip this subject: ", subject)
            print(_sozinds_list)
            print("All sozinds list is empty")
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
            X.append(mat)
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
            X.extend(_X)
            y.extend(_y)
            sozinds_list.extend(_sozinds_list)
            onsetwin_list.extend(_onsetwin_list)
            subjects.extend([subject] * len(_y))

    print(len(X), len(y), len(sozinds_list), len(subjects), len(onsetwin_list))
    return X, y, sozinds_list, onsetwin_list, subjects


def _sequential_aggregation(mat_list, ch_names_list, sozinds_list, outcome):
    X = []
    y = []
    agg_sozinds_list = []
    for mat, ch_names, sozinds in zip(mat_list, ch_names_list, sozinds_list):
        y.append(outcome)
        X.append(mat)
        agg_sozinds_list.append(sozinds)
    return X, y, sozinds_list


def _subsample_matrices_in_time(mat_list):
    maxlen = min([x.shape[1] for x in mat_list])
    if maxlen < 50:
        raise RuntimeError("Preferably not under 50 samples...")

    mat_list = [x[:, :maxlen] for x in mat_list]
    return mat_list


def format_supervised_dataset(
        X,
        sozinds_list,
        onsetwin_list,
        threshold=None,
        window=None,
        weighting_func=None,
        smooth=None,
):
    """Format a supervised learning dataset with (unformatted_X, y).

    This formats unformatted_X to a 4 x T dataset and y is in a set of labels (e.g. 0, or 1 for binary).

    Hyperparameters are:
        - threshold
        - weighting scheme
        - windows chosen
        - smoothing kernel over time

    Parameters
    ----------
    X :
    sozinds_list :
    onsetwin_list :
    threshold :
    window :
    weighting_func :
    smooth :

    Returns
    -------
    newX: np.ndarray
        Stacked data matrix with each heatmap now condensed to four sufficient statistics:
            - mean(SOZ)
            - std(SOZ)
            - mean(SOZ^C)
            - std(SOZ^C)
    """
    newX = []
    dropped_inds = []
    for idx, (data_mat, sozinds, onsetwin) in enumerate(
            zip(X, sozinds_list, onsetwin_list)
    ):
        if onsetwin is None:
            dropped_inds.append(idx)
            print(f"skipping {idx} cuz onsetwin is none")
            continue

        try:
            # apply resampling of the seizure
            data_mat = _resample_seizure(
                data_mat, onsetwin, data_mat.shape[1], desired_length=500
            )
        except Exception as e:
            print(idx, data_mat.shape, np.ptp(sozinds), onsetwin)
            raise Exception(e)

        if smooth is not None:
            # apply moving avg filter
            data_mat = _smooth_matrix(data_mat, window_len=8)

        if threshold is not None:
            data_mat = _apply_threshold(data_mat, threshold=threshold)

        if weighting_func is not None:
            data_mat[:, onsetwin:] = np.apply_along_axis(
                weighting_func, axis=0, arr=data_mat[:, onsetwin:]
            )

        if window is not None:
            window_ = window + onsetwin
            if window_[0] < 0:
                pre_mat = _resample_mat(data_mat[:, :onsetwin], np.abs(window[0]))
                data_mat = np.concatenate((pre_mat, data_mat[:, onsetwin:]), axis=1)
                window_ = window_ + np.abs(window_[0])

            #             print(f"Slicing on window {window} for {data_mat.shape}")
            data_mat = _select_window(data_mat, window_)

        # assemble 4-row dataset
        nsozinds = [i for i in range(data_mat.shape[0]) if i not in sozinds]
        try:
            soz_mat = data_mat[sozinds, :]
            nsoz_mat = data_mat[nsozinds, :]
        except IndexError as e:
            print(idx)
            print(sozinds)
            print(nsozinds)
            print(data_mat.shape)
            print(e)
            continue
            # raise IndexError(e)

        # new_data_mat = np.vstack(
        #     (
        #         np.mean(soz_mat, axis=0),
        #         np.std(soz_mat, axis=0),
        #         np.quantile(soz_mat, q=0.25, axis=0),
        #         np.quantile(soz_mat, q=0.5, axis=0),
        #         np.quantile(soz_mat, q=0.75, axis=0),
        #         np.mean(nsoz_mat, axis=0),
        #         np.std(nsoz_mat, axis=0),
        #         np.quantile(nsoz_mat, q=0.25, axis=0),
        #         np.quantile(nsoz_mat, q=0.5, axis=0),
        #         np.quantile(nsoz_mat, q=0.75, axis=0),
        #     )
        # )

        new_data_mat = np.vstack(
            (
                # np.mean(soz_mat, axis=0),
                # np.std(soz_mat, axis=0),
                np.quantile(soz_mat, q=0.1, axis=0),
                np.quantile(soz_mat, q=0.2, axis=0),
                np.quantile(soz_mat, q=0.3, axis=0),
                np.quantile(soz_mat, q=0.4, axis=0),
                np.quantile(soz_mat, q=0.5, axis=0),
                np.quantile(soz_mat, q=0.6, axis=0),
                np.quantile(soz_mat, q=0.7, axis=0),
                np.quantile(soz_mat, q=0.8, axis=0),
                np.quantile(soz_mat, q=0.9, axis=0),
                np.quantile(soz_mat, q=1.0, axis=0),
                np.quantile(nsoz_mat, q=0.1, axis=0),
                # np.mean(nsoz_mat, axis=0),
                # np.std(nsoz_mat, axis=0),
                np.quantile(nsoz_mat, q=0.2, axis=0),
                np.quantile(nsoz_mat, q=0.3, axis=0),
                np.quantile(nsoz_mat, q=0.4, axis=0),
                np.quantile(nsoz_mat, q=0.5, axis=0),
                np.quantile(nsoz_mat, q=0.6, axis=0),
                np.quantile(nsoz_mat, q=0.7, axis=0),
                np.quantile(nsoz_mat, q=0.8, axis=0),
                np.quantile(nsoz_mat, q=0.9, axis=0),
                np.quantile(nsoz_mat, q=1.0, axis=0),
            )
        )
        newX.append(new_data_mat.reshape(-1, 1).squeeze())

    return np.asarray(newX), dropped_inds


def _evaluate_model(
        clf_func,
        model_params,
        window,
        train_inds,
        X_formatted,
        y,
        groups,
        cv,
        dropped_inds=None,
):
    y = np.array(y).copy().squeeze()
    groups = np.array(groups).copy()
    train_inds = train_inds.copy()

    # if dropped_inds:
    #     for ind in dropped_inds:
    #         # if ind in train_inds:
    #         where_ind = np.where(train_inds >= ind)[0]
    #         train_inds[where_ind] -= 1
    #         train_inds = train_inds[:-1]
    #         # delete index in y, groups
    #         y = np.delete(y, ind)
    #         groups = np.delete(groups, ind)

    # instantiate model
    if clf_func == RandomForestClassifier:
        # instantiate the classifier
        clf = clf_func(**model_params)
    elif clf_func == rerfClassifier:
        model_params.update({"image_width": np.abs(window).sum()})
        clf = clf_func(**model_params)
    else:
        clf = clf_func

    # note that training data (Xtrain, ytrain) will get split again
    Xtrain, ytrain = X_formatted[train_inds, ...], y[train_inds]
    groups_train = groups[train_inds]

    # perform CV using Sklearn
    scoring_funcs = {
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "average_precision": average_precision_score,
    }
    scores = cross_validate(
        clf,
        Xtrain,
        ytrain,
        groups=groups_train,
        cv=cv,
        scoring=list(scoring_funcs.keys()),
        return_estimator=True,
        return_train_score=True,
    )

    return scores


def tune_hyperparameters(
        clf_func,
        unformatted_X,
        y,
        groups,
        train_inds,
        test_inds,
        hyperparameters,
        dataset_params,
        **model_params,
):
    """Perform hyperparameter tuning.

    Pass in X and y dataset that are unformatted yet, and then follow
    a data pipeline that:

    - create the formatted dataset
    - applies hyperparameters
    - cross-validate

    Parameters
    ----------
    clf_func :
    unformatted_X :
    y :
    groups :
    train_inds :
    test_inds :
    hyperparameters :
    dataset_params :
    model_params :

    Returns
    -------
    master_scores: dict
    """

    # CV object to perform training of classifier
    # create Grouped Folds to estimate the mean +/- std performancee
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)

    # track all cross validation score dictionaries
    master_scores = []

    print(f"Using classifier: {clf_func}")
    for idx, hyperparam in enumerate(hyperparameters):
        # extract the hyperparameter explicitly
        window, threshold, weighting_func = hyperparam
        hyperparam_str = (
            f"window-{window}_threshold-{threshold}_weightfunc-{weighting_func}"
        )
        # apply the hyperparameters to the data
        #         print(unformatted_X.shape)
        X_formatted, dropped_inds = format_supervised_dataset(
            unformatted_X,
            **dataset_params,
            window=window,
            threshold=threshold,
            weighting_func=weighting_func,
        )

        scores = _evaluate_model(
            clf_func,
            model_params,
            window,
            train_inds,
            X_formatted,
            y,
            groups,
            cv,
            dropped_inds=dropped_inds,
        )
        # # get the best classifier based on pre-chosen metric
        # best_metric_ind = np.argmax(scores["test_roc_auc"])
        # best_estimator = scores["estimator"][best_metric_ind]
        #
        # # evaluate on the testing dataset
        # X_test, y_test = X_formatted[test_inds, ...], y[test_inds]
        # groups_test = groups[test_inds]
        #
        # y_pred_prob = best_estimator.predict_proba(X_test)[:, 1]
        # y_pred = best_estimator.predict(X_test)
        #
        # # store analysis done on the validation group
        # scores["validate_groups"] = groups_test
        scores["hyperparameters"] = hyperparam
        # scores["validate_ytrue"] = y_test
        # scores["validate_ypred_prob"] = y_pred_prob
        #
        # # pop estimator
        # scores.pop('estimator')
        # scores['estimator'] = best_estimator
        #
        # # resample the held-out test data via bootstrap
        # # test_sozinds_list = dataset_params['sozinds_list'][test_inds]
        # # test_onsetwin_list = dataset_params['onsetwin_list'][test_inds]
        # # X_boot, y_boot, sozinds, onsetwins = resample(X_test, y_test,
        # #                                               test_sozinds_list,
        # #                                               test_onsetwin_list,
        # #                                               n_samples=500)
        #
        # # store ROC curve metrics on the held-out test set
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
        # fnr, tnr, neg_thresholds = roc_curve(y_test, y_pred_prob, pos_label=0)
        # scores["validate_fpr"] = fpr
        # scores["validate_tpr"] = tpr
        # scores["validate_fnr"] = fnr
        # scores["validate_tnr"] = tnr
        master_scores.append(scores)

    return master_scores


def _plot_roc_curve(
        mean_tpr,
        mean_fpr,
        std_tpr=0.0,
        mean_auc=0.0,
        std_auc=0.0,
        label=None,
        ax=None,
        color=None,
        plot_chance=True,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        sns.set_context("paper", font_scale=1.5)
        fig, ax = plt.subplots(1, 1)

    if label is None:
        label = r"Mean ROC (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc)
    else:
        label = fr"{label} (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc)

    if color is None:
        color = "blue"

    # plot the actual curve
    ax.plot(
        mean_fpr, mean_tpr, color=color, label=label, lw=5, alpha=0.8,
    )

    # chance level
    if plot_chance:
        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )

    # get upper and lower bound for tpr
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color=color,
        alpha=0.2,
        # label=r"$\pm$ 1 std. dev.",
    )

    # increase axis limits to see edges
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic curve",
    )
    ax.legend(ncol=2, loc=(1.04, 0))  # "lower right"
    return ax


def load_patient_tfr(deriv_path, subject, band, task=None, verbose=True):
    from mne.time_frequency import read_tfrs, AverageTFR

    if task is not None:
        search_str = f"*sub-{subject}_*task-{task}*.h5"
    else:
        search_str = f"*sub-{subject}_*.h5"

    deriv_files = [f.as_posix() for f in Path(deriv_path).rglob(search_str)]

    if band == "delta":
        fmin, fmax = 0.5, 5
    elif band == "theta":
        fmin, fmax = 5, 10
    elif band == "alpha":
        fmin, fmax = 10, 16
    elif band == "beta":
        fmin, fmax = 16, 30
    elif band == "gamma":
        fmin, fmax = 30, 90
    elif band == "highgamma":
        fmin, fmax = 90, 300
    else:
        raise ValueError("kwarg 'band' can only be prespecified set of values")

    # print(deriv_files)
    patient_tfrs = []
    for deriv_fpath in deriv_files:
        # print(f"Loading {deriv_fpath}")
        avg_tfr = read_tfrs(deriv_fpath)[0]

        # print(avg_tfr.freqs)
        # only obtain the Band TFR for that subject
        freq_inds = np.where((avg_tfr.freqs >= fmin) & (avg_tfr.freqs < fmax))[0]
        # print(freq_inds)
        band_data = np.mean(avg_tfr.data[:, freq_inds, :], axis=1, keepdims=True)
        band_tfr = AverageTFR(
            avg_tfr.info,
            data=band_data,
            freqs=[fmin],
            nave=1,
            times=avg_tfr.times,
            verbose=0,
        )

        json_fpath = deriv_fpath.replace(".h5", ".json")
        with open(json_fpath, "r") as fin:
            sidecar_json = json.load(fin)

        # obtain the event IDs for the markers of interest
        sz_onset_id = sidecar_json.get("sz_onset_event_id", None)
        sz_offset_id = sidecar_json.get("sz_offset_event_id")
        clin_onset_id = sidecar_json.get("clin_onset_event_id")

        # events array
        events = sidecar_json["events"]

        # obtain onset/offset event markers
        sz_onset_win = _get_onset_event_id(events, sz_onset_id)
        sz_offset_win = _get_onset_event_id(events, sz_offset_id)
        clin_onset_win = _get_onset_event_id(events, clin_onset_id)

        # set those windows
        sidecar_json["sz_onset_win"] = sz_onset_win
        sidecar_json["sz_offset_win"] = sz_offset_win
        sidecar_json["clin_onset_win"] = clin_onset_win
        sidecar_json["freq_band"] = (fmin, fmax)

        # create a Result object
        band_tfr = Result(
            Normalize.compute_fragilitymetric(band_data.squeeze(), invert=True),
            info=avg_tfr.info,
            metadata=sidecar_json,
        )
        # band_tfr.metadata.save(json_fpath)

        if np.isnan(band_data).any():
            print(f"Skipping {deriv_fpath} due to nans")
            continue

        if sz_onset_win is None:
            print(f"Skipping {deriv_fpath}")
            continue

        patient_tfrs.append(band_tfr)
    return patient_tfrs


def load_patient_graphstats(
        deriv_path, subject, kind="ieeg", task="ictal", band=None, verbose=True
):
    from eztrack.io.read_result import read_result_eztrack

    if band is not None:
        if band == "delta":
            fmin, fmax = 0.5, 4
        elif band == "theta":
            fmin, fmax = 4, 8
        elif band == "alpha":
            fmin, fmax = 8, 13
        elif band == "beta":
            fmin, fmax = 13, 30
        elif band == "gamma":
            fmin, fmax = 30, 90
        elif band == "highgamma":
            fmin, fmax = 90, 300
        else:
            raise ValueError("kwarg 'band' can only be prespecified set of values")
        search_str = f"*sub-{subject}_*task-{task}*freq-{band}_{kind}.npz"
    else:
        search_str = f"*sub-{subject}_*task-{task}*{kind}.npz"

    deriv_files = [f.as_posix() for f in Path(deriv_path).rglob(search_str)]

    patient_graphstats = []
    for deriv_fpath in deriv_files:
        deriv_basename = os.path.basename(deriv_fpath)
        if "freq" in deriv_basename:
            deriv_basename = deriv_basename.split("_freq")[0]
        gs_result = read_result_eztrack(
            deriv_path=deriv_path,
            deriv_fname=deriv_basename,
            numpy_str="conn_mats",
            normalize=True,
            verbose=verbose,
        )

        if gs_result.get_metadata()["sz_onset_win"] is None:
            print(f"Skipping {deriv_fpath}")
            continue

        patient_graphstats.append(gs_result)
    return patient_graphstats


def _get_feature_deriv_path(deriv_path, feature_name):
    freq_bands = ["delta", "theta", "alpha", "beta", "gamma", "highgamma"]
    coh_degree_band_names = [f"{band}-coherence-degree" for band in freq_bands]
    coh_cent_band_names = [f"{band}-coherence-centrality" for band in freq_bands]

    ext = ".npz"

    # get all files inside experiment
    if feature_name in freq_bands:
        deriv_path = deriv_path / "tfr" / "average"
        ext = ".h5"
    elif feature_name in ["correlation-centrality", "correlation-degree"]:
        deriv_path = deriv_path / feature_name / "average"

    elif feature_name in coh_degree_band_names:
        band = feature_name.split("-")[0]
        deriv_path = deriv_path / "coherence-degree" / "average" / band

    elif feature_name in coh_cent_band_names:
        band = feature_name.split("-")[0]
        deriv_path = deriv_path / "coherence-centrality" / "average" / band
    return deriv_path, ext


def load_patient_dict(deriv_path, feature_name, task=None, subjects=None, verbose=True):
    """Load comparative features patient dictionary of results."""
    deriv_path = Path(deriv_path)
    patient_result_dict = collections.defaultdict(list)

    freq_bands = ["delta", "theta", "alpha", "beta", "gamma", "highgamma"]

    # get path to this specific feature and its extension used
    feature_deriv_path, ext = _get_feature_deriv_path(deriv_path, feature_name)
    if subjects is None:
        subjects = [
            fpath.name for fpath in feature_deriv_path.glob("*") if fpath.is_dir()
        ]
    if any([band in feature_name for band in freq_bands]):
        band = feature_name.split("-")[0]
    else:
        band = None

    if verbose:
        print(f"Loading data from: {feature_deriv_path}")
        print(subjects)
        print(band, task, feature_name)

    for subject in subjects:
        if feature_name in freq_bands:
            patient_result_dict[subject] = load_patient_tfr(
                feature_deriv_path, subject, feature_name, task=task,
            )
        else:
            patient_result_dict[subject] = load_patient_graphstats(
                feature_deriv_path,
                subject,
                kind="ieeg",
                band=band,
                task=task,
                verbose=False,
            )
    # subj_results = Parallel(n_jobs=num_cores)(
    #     delayed(_load_features)(
    #         feature_name, subject, feature_deriv_path, freq_bands, band, task
    #     )
    #     for _, subject in enumerate(tqdm(subjects))
    # )
    # print(len(subj_results))
    # print(subj_results[0])
    # # transform list of dicts to dict
    # patient_result_dict = {
    #     subject: results for x in subj_results for subject, results in x.items()
    # }

    if verbose:
        print("Got ", len(patient_result_dict), " patients")
        print(patient_result_dict.keys())
    return patient_result_dict


def _load_features(feature_name, subject, feature_deriv_path, freq_bands, band, task):
    patient_result_dict = dict()
    if feature_name in freq_bands:
        patient_result_dict[subject] = load_patient_tfr(
            feature_deriv_path, subject, feature_name, task=task,
        )
    else:
        patient_result_dict[subject] = load_patient_graphstats(
            feature_deriv_path, subject, kind="ieeg", band=band, task=task,
        )
    return patient_result_dict


def summarize_feature_comparisons(
        base_clf: BaseEstimator, comparison_clfs: Dict[str, BaseEstimator], X_test, y_test
):
    from mlxtend.evaluate import mcnemar, cochrans_q, mcnemar_table

    summary_dict = collections.OrderedDict()
    mcnemar_tbs = dict()

    # create list of predicted values
    base_y_predict = base_clf.predict(X_test)
    y_predictions = [base_y_predict]
    for idx, (name, clf) in enumerate(comparison_clfs.items()):
        # get the probability
        y_predict_proba = clf.predict_proba(X_test)
        y_predict = clf.predict(X_test)

        # form mcnemar tables against base classifier
        tb = mcnemar_table(y_test, base_y_predict, y_predict)
        mcnemar_tbs[f"base vs {name}"] = tb.values()

        # store predictions per classifier
        y_predictions.append(y_predict)

    # first run cochrans Q test
    qstat, pval = cochrans_q(y_test, *y_predictions)
    summary_dict["cochrans_q"] = qstat
    summary_dict["cochrans_q_pval"] = pval

    # run mcnemars test against all the predictions
    for name, table in mcnemar_tbs.items():
        chi2stat, pval = mcnemar(table, exact=True)
        summary_dict[f"mcnemar_{name}_chi2stat"] = chi2stat
        summary_dict[f"mcnemar_{name}_pval"] = pval

    return summary_dict


def check_mcnemar_significance(mcnemar_pvals):
    import pingouin as pg

    reject, pvals = pg.multicomp(mcnemar_pvals, alpha=0.05, method="holm")

    return reject, pvals


def compute_acc_with_ci(clf, X_test, y_test):
    from mlxtend.evaluate import bootstrap_point632_score
    from sklearn.metrics import balanced_accuracy_score

    scores = bootstrap_point632_score(
        clf,
        X=X_test,
        y=y_test,
        n_splits=500,
        method=".632+",
        clone_estimator=True,
        scoring_func=balanced_accuracy_score,
    )
    return scores


def determine_feature_importances(clf, X, y, n_jobs):
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        estimator=clf,
        X=X,
        y=y,
        scoring="roc_auc",
        n_repeats=5,
        n_jobs=n_jobs,
        random_state=1,
    )

    std = result.importances_std
    indices = np.argsort(result.importances_mean)[::-1]

    return result


def compute_auc_optimism(clf, X, y, n_boot=500, alpha=0.05):
    from sklearn.utils import resample
    from sklearn.metrics import roc_auc_score

    # original auc
    orig_metric = roc_auc_score(y, clf.predict_proba(X))

    score_biases = []

    for boot_idx in range(n_boot):
        boot_X, boot_y = resample(X, y, replace=True, n_samples=len(y), stratify=y)
        clf.fit(boot_X)

        # bootstrap sample score
        y_predict_proba = clf.predict_proba(boot_X)
        C_boot_roc_auc = roc_auc_score(boot_y, y_predict_proba)

        # original sample score
        y_predict_proba = clf.predict_proba(X)
        C_orig_roc_auc = roc_auc_score(y, y_predict_proba)

        # store the bias
        score_biases.append(C_boot_roc_auc - C_orig_roc_auc)

    # compute CI
    lb = np.percentile(score_biases, q=alpha // 2)
    ub = np.percentile(score_biases, q=1 - alpha // 2)

    # compute optimism
    optimism = np.mean(score_biases)
    ci = [lb, ub]
    return orig_metric - optimism, ci


def _show_calibration_curve(estimators, X, y, name):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib.pyplot as plt
    import seaborn as sns

    #
    sns.set_context("paper", font_scale=1.5)
    plt.figure(figsize=(7, 10))
    ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    clf = estimators[0]
    y_predict_prob = clf.predict_proba(X)
    prob_pos = y_predict_prob[:, 1]
    # compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y, prob_pos, n_bins=100, strategy="quantile"
    )

    clf_score = np.round(brier_score_loss(y, prob_pos, pos_label=np.array(y).max()), 2)

    print(clf_score)
    print(fraction_of_positives, mean_predicted_value)

    # frac_pred_vals = []
    # mean_pred_values = np.linspace(0, 1.0, 200)
    # brier_scores = []
    # for i, clf in enumerate(estimators):
    #     y_predict_prob = clf.predict_proba(X)
    #     prob_pos = y_predict_prob[:, 1]
    #     # compute calibration curve
    #     fraction_of_positives, mean_predicted_value = calibration_curve(
    #         y, prob_pos, n_bins=10, strategy="quantile"
    #     )
    #
    #     clf_score = np.round(
    #         brier_score_loss(y, prob_pos, pos_label=np.array(y).max()), 2
    #     )
    #
    #     # create a linear interpolation of the calibration
    #     interp_frac_positives = np.interp(
    #         mean_pred_values, mean_predicted_value, fraction_of_positives
    #     )
    #     interp_frac_positives[0] = 0.0
    #
    #     # store curves + scores
    #     brier_scores.append(clf_score)
    #     frac_pred_vals.append(interp_frac_positives)
    #
    # mean_frac_pred_values = np.mean(frac_pred_vals, axis=0)
    # ax1.plot(
    #     mean_pred_values,
    #     mean_frac_pred_values,
    #     "s-",
    #     label=rf"{name.capitalize()} ({np.round(np.mean(brier_scores),2)} $\pm$ {np.round(np.std(brier_scores), 2)}",
    # )
    #
    # # get upper and lower bound for tpr
    # std_fpv = np.std(frac_pred_vals, axis=0)
    # tprs_upper = np.minimum(mean_frac_pred_values + std_fpv, 1)
    # tprs_lower = np.maximum(mean_frac_pred_values - std_fpv, 0)
    # ax1.fill_between(
    #     mean_pred_values,
    #     tprs_lower,
    #     tprs_upper,
    #     color="grey",
    #     alpha=0.2,
    #     # label=r"$\pm$ 1 std. dev.",
    # )

    # actually do the plot
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label=f"{name.capitalize()} ({clf_score})",
    )

    # set
    ax1.plot()
    ax1.set(
        ylabel="Fraction of Success Outcomes (y label of 1)",
        xlabel="Mean predicted confidence statistic",
        ylim=[-0.05, 1.05],
        title="Calibration plots  (reliability curve)",
    )
    ax1.legend(loc="lower right")
    return ax1


def show_calibration_curves(clfs, X, y):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib.pyplot as plt
    import seaborn as sns

    #
    sns.set_context("paper", font_scale=1.5)
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, clf in clfs.items():
        print(name, clf)
        y_predict_prob = clf.predict_proba(X)
        prob_pos = y_predict_prob[:, 1]
        # compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, prob_pos, n_bins=10, strategy="quantile"
        )

        clf_score = brier_score_loss(y, prob_pos, pos_label=y.max())

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"{name} ({clf_score})",
        )

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.plot()
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots  (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.show()
