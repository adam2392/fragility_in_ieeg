from pathlib import Path

from eztrack import (
    preprocess_raw,
    lds_raw_fragility,
    write_result_fragility,
    plot_result_heatmap,
)
from mne.utils import warn
from mne_bids import read_raw_bids, BIDSPath, get_entity_vals


def _append_path(path, resample_sfreq, subject, sfreq):
    if resample_sfreq:
        sample_rate_name = f"resampled-{sfreq}Hz"
    else:
        sample_rate_name = f"{sfreq}Hz"

    return (path
            / sample_rate_name
            / "fragility"
            / reference
            / f"sub-{subject}")


def run_analysis(
        bids_path, reference="monopolar", resample_sfreq=None, deriv_path=None,
        figures_path=None, excel_fpath=None, verbose=True, overwrite=False,
):
    subject = bids_path.subject

    # load in the data
    raw = read_raw_bids(bids_path)
    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False)
    raw.load_data()

    if resample_sfreq:
        if resample_sfreq > raw.info['sfreq']:
            return

        # perform resampling
        raw = raw.resample(resample_sfreq, n_jobs=-1)

    if deriv_path is None:
        deriv_path = (
                bids_path.root
                / "derivatives"
        )

    sfreq_int = int(raw.info['sfreq'])
    deriv_path = _append_path(deriv_path, resample_sfreq=resample_sfreq,
                              subject=subject, sfreq=sfreq_int)
    # set where to save the data output to
    if figures_path is None:
        figures_path = (
                bids_path.root
                / "derivatives"
                / "figures"
        )
    if resample_sfreq:
        sample_rate_name = f"resampled-{sfreq}Hz"
    else:
        sample_rate_name = f"{sfreq}Hz"
    deriv_root = (figures_path
                  # / 'nodepth'
                  / sample_rate_name
                  / "raw" \
                  / reference \
                  / f"sub-{subject}")
    figures_path = _append_path(figures_path, resample_sfreq=resample_sfreq,
                                subject=subject, sfreq=sfreq_int)

    # use the same basename to save the data
    deriv_basename = bids_path.basename
    bids_entities = bids_path.entities
    deriv_basename_nosuffix = BIDSPath(**bids_entities).basename
    print(deriv_basename_nosuffix)
    if len(list(deriv_path.rglob(f'{deriv_basename_nosuffix}*.npy'))) > 0 and not overwrite:
        warn(f'The {deriv_basename}.npy exists, but overwrite if False.')
        return

    # pre-process the data using preprocess pipeline
    datatype = bids_path.datatype
    print('Power Line frequency is : ', raw.info["line_freq"])
    raw = preprocess_raw(raw, datatype=datatype,
                         verbose=verbose, method="simple", drop_chs=False)

    # plot raw data
    deriv_root.mkdir(exist_ok=True, parents=True)
    fig_basename = bids_path.copy().update(extension='.pdf').basename
    scale = 200e-6
    fig = raw.plot(
        scalings={
            'ecog': scale,
            'seeg': scale
        }, n_channels=len(raw.ch_names))
    fig.savefig(deriv_root / fig_basename)

    # raise Exception('hi')
    model_params = {
        "winsize": 500,
        "stepsize": 250,
        "radius": 1.5,
        "method_to_use": "pinv",
    }
    # run heatmap
    result, A_mats, delta_vecs_arr = lds_raw_fragility(
        raw, reference=reference, return_all=True, **model_params
    )

    # write results to
    result_sidecars = write_result_fragility(
        A_mats,
        delta_vecs_arr,
        result=result,
        deriv_basename=deriv_basename,
        deriv_path=deriv_path,
        verbose=verbose,
    )
    fig_basename = deriv_basename

    result.normalize()
    # create the heatmap
    plot_result_heatmap(
        result=result,
        fig_basename=fig_basename,
        figures_path=figures_path,
        excel_fpath=excel_fpath
    )


if __name__ == "__main__":
    WORKSTATION = "lab"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        # the root of the BIDS dataset
        root = Path("/Users/adam2392/Dropbox/epilepsy_bids/")
        output_dir = root / 'derivatives'

        figures_dir = output_dir / 'figures'

        # path to excel layout file - would be changed to the datasheet locally
        excel_fpath = Path(
            "/Users/adam2392/Dropbox/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
        )
    elif WORKSTATION == "lab":
        root = Path("/home/adam2392/hdd/epilepsy_bids/")
        excel_fpath = Path(
            "/home/adam2392/hdd/epilepsy_bids/sourcedata/organized_clinical_datasheet_raw.xlsx"
        )

        # output directory
        output_dir = Path("/home/adam2392/hdd2") / 'derivatives'

        # figures directory
        figures_dir = output_dir / 'figures'

    # define BIDS entities
    # SUBJECTS = [
    #     # 'pt1', 'pt2', 'pt3',  # NIH
    #     'jh103', 'jh105',  # JHH
    #     'umf001', 'umf002', 'umf003', 'umf005',  # UMF
    #     #     # 'la00', 'la01', 'la02', 'la03', 'la04', 'la05', 'la06',
    #     #     # 'la07'
    # ]

    session = "presurgery"  # only one session
    task = "ictal"
    datatype = "ieeg"
    acquisition = "ecog"  # or SEEG
    extension = ".vhdr"

    # analysis parameters
    reference = 'average'
    sfreq = 500

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")

    # perform loop over datasets
    for subject in all_subjects:
        # if subject not in SUBJECTS:
        #     continue
        ignore_subs = [sub for sub in all_subjects if sub != subject]
        all_tasks = get_entity_vals(root, "task", ignore_subjects=ignore_subs)
        ignore_tasks = [tsk for tsk in all_tasks if tsk != task]

        print(f"Analyzing {task} task.")
        ignore_tasks = [tsk for tsk in all_tasks if tsk != task]
        runs = get_entity_vals(
            root, 'run', ignore_subjects=ignore_subs,
            ignore_tasks=ignore_tasks,
            ignore_acquisitions=['seeg']
        )
        print(f'Found {runs} runs for {task} task.')

        for idx, run in enumerate(runs):
            # create path for the dataset
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                run=run,
                datatype=datatype,
                acquisition=acquisition,
                suffix=datatype,
                root=root,
                extension=extension,
            )
            print(f"Analyzing {bids_path}")

            run_analysis(bids_path, reference=reference,
                         resample_sfreq=sfreq,
                         deriv_path=output_dir, figures_path=figures_dir,
                         excel_fpath=excel_fpath)
