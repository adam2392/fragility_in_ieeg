"""API for running fragility visualization."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import mne
import pandas as pd
from eztrack import (
    read_derivative
)
from eztrack.base.statistics.sampling import resample_seizure
from eztrack.base.utils.file_utils import _replace_ext
from eztrack.io import read_result_eztrack, read_clinical_excel
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from eztrack.viz.plot_fragility_ts import plot_fragility_ts
from mne.viz import plot_alignment, snapshot_brain_montage
from mne_bids import BIDSPath, get_entity_vals
from mne_bids.path import get_entities_from_fname, _parse_ext
import matplotlib.animation as animation
plt.ion()

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


def run_plot_heatmap(deriv_path, figures_path):
    deriv = read_derivative(deriv_path)

    # read in channel types
    bids_path = BIDSPath(**deriv.info.source_entities,
                         datatype=deriv.info['datatype'],
                         root=deriv.bids_root)
    bids_path.update(suffix='channels', extension='.tsv')

    # read in sidecar channels.tsv
    channels_pd = pd.read_csv(bids_path.fpath, sep='\t')
    description_chs = pd.Series(channels_pd.description.values, index=channels_pd.name).to_dict()
    print(description_chs)
    resected_chs = [ch for ch, description in description_chs.items() if description == 'resected']
    print(f'Resected channels are {resected_chs}')

    # set title name as the filename
    title = Path(deriv_path).stem

    figure_fpath = Path(figures_path) / Path(deriv_path).with_suffix('.pdf').name

    # normalize
    deriv.normalize()
    cbarlabel = 'Fragility'

    subject = 'fsaverage'
    subjects_dir = '/Users/adam2392/Dropbox/epilepsy_bids/derivatives/freesurfer'
    electrodes_fpath = '/Users/adam2392/Downloads/nihcoords/NIH040.csv'
    elec_df = pd.read_csv(electrodes_fpath)
    ch_names = elec_df['chanName'].tolist()
    ch_coords = elec_df[['x', 'y', 'z']].to_numpy(dtype=float) / 1000.
    ch_pos = dict(zip(ch_names, ch_coords))
    print(ch_pos)

    lpa, nasion, rpa = mne.coreg.get_mni_fiducials(
        subject, subjects_dir=subjects_dir)
    lpa, nasion, rpa = lpa['r'], nasion['r'], rpa['r']

    montage = mne.channels.make_dig_montage(
        ch_pos, coord_frame='mri', nasion=nasion, lpa=lpa, rpa=rpa)
    print('Created %s channel positions' % len(ch_names))

    trans = mne.channels.compute_native_head_t(montage)
    print(trans)

    info = mne.create_info(ch_names=deriv.ch_names, sfreq=8.0)
    raw = mne.io.RawArray(deriv.get_data(), info)
    # attach montage
    raw.set_montage(montage)

    fig = plot_alignment(raw.info, subject=subject, subjects_dir=subjects_dir,
                         surfaces=['pial'], trans=trans, coord_frame='mri')
    mne.viz.set_3d_view(fig, 200, 70, focalpoint=[0, -0.005, 0.03])

    xy, im = snapshot_brain_montage(fig, montage, hide_sensors=False)

    # Convert from a dictionary to array to plot
    xy_pts = np.vstack([xy[ch] for ch in raw.info['ch_names']])

    # create an initialization and animation function
    # to pass to FuncAnimation
    def init():
        """Create an empty frame."""
        return paths,

    def animate(i, activity):
        """Animate the plot."""
        paths.set_array(activity[:, i])
        return paths,
    cmap = 'turbo'
    vmin = 0
    vmax = 0.7
    # create the figure and apply the animation of the
    # gamma frequency band activity
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)
    ax.set_axis_off()
    paths = ax.scatter(*xy_pts.T, c=np.zeros(len(xy_pts)), s=200,
                       cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(paths, ax=ax)
    ax.set_title('Gamma frequency over time (Hilbert transform)',
                 size='large')

    # avoid edge artifacts and decimate, showing just a short chunk
    show_power = raw.get_data()[:, 100:700:2]
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   fargs=(show_power,),
                                   frames=show_power.shape[1],
                                   interval=100, blit=True)
    plt.show(block=True)
    plt.waitforbuttonpress()
    # plt.pause(0.001)
    # raise Exception('hi')
    # run heatmap plot
    # deriv.plot_heatmap(
    #     cmap='turbo',
    #     cbarlabel=cbarlabel,
    #     title=title,
    #     figure_fpath=figure_fpath,
    #     soz_chs=resected_chs,
    # )
    print(f'Saved figure to {figure_fpath}')


if __name__ == "__main__":
    # the root of the BIDS dataset
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        root = Path('/Users/adam2392/Dropbox/epilepsy_bids/')
        source_dir = root / 'sourcedata'
        deriv_root = root / 'derivatives' / 'interictal'

        figures_root = deriv_root / 'figures'

        # path to excel layout file - would be changed to the datasheet locally
        excel_fpath = source_dir / 'organized_clinical_datasheet_raw.xlsx'
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

    # define BIDS entities
    SUBJECTS = [
        'pt1'
    ]

    # pre, Sz, Extraoperative, post
    task = "interictal"
    acquisition = "ecog"
    datatype = "ieeg"
    extension = ".vhdr"
    session = "presurgery"  # only one session

    # analysis parameters
    reference = 'average'
    sfreq = None

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")
    for subject in all_subjects:
        if subject not in SUBJECTS:
            continue
        ignore_subs = [sub for sub in all_subjects if sub != subject]

        # get all sessions
        ignore_set = {
            'ignore_subjects': ignore_subs,
        }
        print(f'Ignoring these sets: {ignore_set}')
        all_tasks = get_entity_vals(root, "task", **ignore_set)
        tasks = all_tasks
        tasks = ['interictal']

        for task in tasks:
            print(f"Analyzing {task} task.")
            ignore_tasks = [tsk for tsk in all_tasks if tsk != task]
            ignore_set['ignore_tasks'] = ignore_tasks
            runs = get_entity_vals(
                root, 'run', **ignore_set
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

                deriv_basename = _add_desc_to_bids_fname(bids_path.basename,
                                                         description=DERIVATIVETYPES.COLPERTURB_MATRIX.value)
                deriv_chain = Path('originalsampling') / "fragility" / reference / f"sub-{subject}"
                deriv_path = deriv_root / deriv_chain / deriv_basename
                figures_path = figures_root / deriv_chain
                run_plot_heatmap(deriv_path=deriv_path, figures_path=figures_path)

                # run plot raw data
                # figures_path = figures_root / Path('originalsampling') / "raw" / reference / f"sub-{subject}"
                # run_plot_raw(bids_path=bids_path, resample_sfreq=None, figures_path=figures_path)
