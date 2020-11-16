"""
This script extracts datasets into a `.npz` file in order to facilitate faster
IO for the results leading to the figures in the manuscript.
"""
from pathlib import Path

import numpy as np
import mne

mne.set_log_level("ERROR")

from eztrack.base.publication.study import load_patient_dict, extract_Xy_pairs

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
    "jh107",
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

# BIDS related directories
bids_root = Path("/Volumes/Seagate Portable Drive/data")
bids_root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
bids_root = Path("/home/adam2392/hdd2/Dropbox/epilepsy_bids/")

deriv_path = "/Users/adam2392/Dropbox/epilepsy_bids/derivatives/"
deriv_path = "/home/adam2392/hdd/derivatives/"

# BIDS entities
session = "presurgery"
acquisition = "seeg"
task = "ictal"
kind = "ieeg"
reference = "average"
patient_aggregation_method = None

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

for feature_name in feature_names:
    feature_subject_dict = load_patient_dict(
        deriv_path / "openneuro" / "derivatives",
        feature_name,
        task="ictal",
        subjects=subjects,
    )

    # get the (X, y) tuple pairs
    unformatted_X, y, sozinds_list, onsetwin_list, subject_groups = extract_Xy_pairs(
        feature_subject_dict,
        excel_fpath=excel_fpath,
        patient_aggregation_method=patient_aggregation_method,
        verbose=False,
    )
    # print(unformatted_X[0])
    # break
    print(
        len(unformatted_X),
        len(y),
        len(subject_groups),
        len(onsetwin_list),
        len(sozinds_list),
    )
    fpath = (
        Path(deriv_path).parent / "baselinesliced" / f"{feature_name}_unformatted.npz"
    )
    fpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fpath,
        unformatted_X=unformatted_X,
        y=y,
        sozinds_list=sozinds_list,
        onsetwin_list=onsetwin_list,
        subject_groups=subject_groups,
        subjects=subjects,
    )
