"""API for running fragility visualization."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from mne_bids.path import get_entities_from_fname, _parse_ext
from natsort import natsorted

from eztrack.base.statistics.sampling import resample_seizure
from eztrack.base.utils.file_utils import _replace_ext
from eztrack.io import read_result_eztrack, read_clinical_excel
from eztrack.preprocess.bids_tree import get_bids_layout
from eztrack.utils.config import BIDS_LAYOUT_DB_PATH
from eztrack.viz import plot_result_heatmap
from eztrack.viz.plot_fragility_ts import plot_fragility_ts


def _generate_fragility_ts(
    deriv_path, deriv_fname, fig_path, excel_fpath=None, overwrite=False
):
    # run visualization automatically
    fig_path = Path(fig_path)
    fig_path.mkdir(exist_ok=True, parents=True)

    # figure filename
    fig_fname = _replace_ext(deriv_fname, ".pdf", verbose=verbose)

    # create figure for SOZ
    fig_fpath = fig_path / fig_fname

    if not overwrite and fig_fpath.exists():
        print(f"{fig_fpath} already exists!")
        return

    # set derivatives path and temporary directory to cache results
    params = get_entities_from_fname(deriv_fname)
    subject = params["sub"]

    # load in the actual data
    if not deriv_fname.endswith(".json"):
        fname, ext = _parse_ext(deriv_fname, verbose=verbose)
        json_fname = fname + ".json"
    else:
        json_fname = deriv_fname
    result = read_result_eztrack(
        deriv_path, deriv_fname=json_fname, normalize=True, verbose=verbose
    )
    metadata = result.get_metadata()
    ch_names = metadata["ch_names"]

    # get SOZ, RZ and their complement  set  of channels
    if excel_fpath is not None:
        pat_dict = read_clinical_excel(excel_fpath, subject=subject)
        soz_chs = pat_dict["SOZ_CONTACTS"]
        rz_chs = pat_dict["RESECTED_CONTACTS"]
        if not rz_chs:
            rz_chs = pat_dict["ABLATED_CONTACTS"]

        outcome = pat_dict["OUTCOME"]
        ilae = pat_dict["ILAE_SCORE"]
        engel = pat_dict["ENGEL_SCORE"]
        cc = pat_dict["CLINICAL_COMPLEXITY"]

        soz_inds = [ind for ind, ch in enumerate(ch_names) if ch in soz_chs]
        nsoz_inds = [ind for ind in range(len(ch_names)) if ind not in soz_inds]
        rz_inds = [ind for ind, ch in enumerate(ch_names) if ch in rz_chs]
        nrz_inds = [ind for ind in range(len(ch_names)) if ind not in rz_inds]

    sz_onset_win = result.get_metadata()["sz_onset_win"]
    if result.get_metadata()["sz_offset_win"] is None:
        sz_offset_win = len(result)
        result.metadata["sz_offset_win"] = sz_offset_win

    # re-sample seizure
    result = resample_seizure(result, desired_len=500, verbose=True)

    # trim around the seizure (-80 -> entire seizure)
    fragmat = result.get_data(
        start=sz_onset_win - 80, stop=result.get_metadata()["sz_offset_win"]
    )

    fig, axs = plt.subplots(2, 1, figsize=(15, 8))
    title = (
        f"Fragility Over Time \n "
        f"Outcome: {outcome}, ILAE: {ilae}, Engel: {engel} \n"
        f"Clinical Complexity: {cc}"
    )
    _, ax = plot_fragility_ts(
        fragmat=fragmat,
        ch_inds=[soz_inds, nsoz_inds],
        labels=["SOZ", "$SOZ^C$"],
        colors=["red", "black"],
        title=title,
        output_fpath=fig_fpath,
        ax=axs.flat[0],
    )
    ax.axvline(80, lw=3, ls="--", color="black", label="SZ Onset")

    # create figure for RZ
    _, ax = plot_fragility_ts(
        fragmat=fragmat,
        ch_inds=[rz_inds, nrz_inds],
        labels=["RZ", "$RZ^C$"],
        colors=["red", "black"],
        title=title,
        output_fpath=fig_fpath,
        ax=axs.flat[1],
    )
    ax.axvline(80, lw=3, ls="--", color="black", label="SZ Onset")
    ax.legend()
    fig.savefig(fig_fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure at {fig_fpath}")


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
        deriv_path = Path("/home/adam2392/hdd/openneuro/derivatives")

        # figures directory
        figures_dir = Path(
            "/home/adam2392/hdd2/epilepsy_bids/derivatives/fragility/figures"
        )

    elif WORKSTATION == "test":
        bids_root = Path("/home/adam2392/Documents/eztrack/data/bids_layout/")

    centers = [
        # "ummc",
        # "jhu",
        # "nih",
        # "umf",
        "cleveland",
        "clevelandnl",
        "clevelandtvb",
    ]
    # BIDS-identifier variables to change
    acquisition = "seeg"
    kind = "ieeg"
    task = "ictal"
    session = "presurgery"
    reference = "monopolar"
    verbose = True
    overwrite = True

    subjects = {}
    soz_subs = {}

    layout = get_bids_layout(bids_root, database_path=BIDS_LAYOUT_DB_PATH)

    for center in centers:
        # center = "cleveland"
        sourcedir = Path(source_dir / center)
        subject_ids = [os.path.basename(x) for x in sourcedir.glob("*") if x.is_dir()]

        # get all subject ids
        # subject_ids = [sub for sub in layout.get_subjects()
        # subject_ids = layout.get()
        print(sourcedir)
        print("Got subject ids: ", subject_ids)

        for subject in subject_ids:
            # get all the subject runs
            # deriv_path = Path(
            #     output_dir / "derivatives" / "fragility" / reference / subject
            # )
            fig_path = Path(figures_dir) / reference

            if subject in ["umf006"]:
                continue
            # if subject == "nl03":
            #    continue

            _fnames = natsorted(
                [
                    x.name
                    for x in (
                        deriv_path / "fragility" / reference / f"sub-{subject}"
                    ).rglob(f"*_{kind}.json")
                    # if subject in x.name
                    if session == get_entities_from_fname(x.name)["session"]
                    if task == get_entities_from_fname(x.name)["task"]
                    # if reference in x.name
                ]
            )
            print("Found derivatives files: ", _fnames)

            # run analysis
            for deriv_fname in _fnames:
                # for x in (deriv_path / 'fragility').rglob("*.npz"):
                #     print(x.name)
                # if not any([output_fname in os.path.basename(x.name) for x in deriv_path.rglob("*.npz")]):
                #     print(f'{output_fname} not inside derivatives directory. Compute first. Skipping...')
                #     continue
                print(f"Analyzing bids filename: {deriv_fname}")

                # read in the result
                result = read_result_eztrack(
                    deriv_path / "fragility" / reference, deriv_fname=deriv_fname
                )

                plot_result_heatmap(
                    result=result,
                    fig_basename=deriv_fname,
                    figures_path=fig_path,
                    # excel_fpath=excel_fpath
                )
                # _generate_fragility_ts(
                #     deriv_path,
                #     deriv_fname,
                #     fig_path,
                #     excel_fpath=excel_fpath,
                #     overwrite=overwrite,
                # )
