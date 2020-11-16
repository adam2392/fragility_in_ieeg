"""API for running fragility visualization."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
import mne
import numpy as np
import pandas as pd
from eztrack.io import (
    read_derivative_npy
)
from eztrack.base.statistics.sampling import resample_seizure
from eztrack.base.utils.file_utils import _replace_ext
from eztrack.io import read_clinical_excel
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from eztrack.viz.plot_fragility_ts import plot_fragility_ts
from mayavi import mlab
from mne.viz import plot_alignment, snapshot_brain_montage
from mne_bids import BIDSPath, get_entity_vals
from mne_bids.path import get_entities_from_fname, _parse_ext


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
    result = read_derivative_npy(
        deriv_path
    )
    metadata = result.info
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


def run_plot_brain(deriv_path, figures_path, root, show_alignment: bool=False):
    # subject = 'fsaverage'
    figdir = Path('/Users/adam2392/Dropbox/epilepsy_bids/derivatives/interictal/figures')
    subjects_dir = '/Users/adam2392/Dropbox/epilepsy_bids/derivatives/freesurfer'
    subject = 'fsaverage'

    # read in dataset
    deriv = read_derivative_npy(Path(deriv_path).with_suffix('.npy'))
    deriv.load_data()
    deriv.normalize()

    # set bids root for derivative
    deriv.info['root'] = root

    # read in channel montage from the raw data
    bids_path = BIDSPath(**deriv.info.source_entities,
                         datatype=deriv.info['datatype'],
                         root=deriv.bids_root)
    # subject = bids_path.subject
    if not (Path(subjects_dir) / bids_path.subject / 'bem').exists():
        from mne.bem import make_watershed_bem
        make_watershed_bem(subject=bids_path.subject,
                           subjects_dir=subjects_dir,
                           brainmask="../../mri/brainmask.mgz")

    print(bids_path)
    # raw = read_raw_bids(bids_path)
    # montage = raw.get_montage()

    # find electrodes file
    elec_path = BIDSPath(subject=bids_path.subject, session=bids_path.session,
                         space='fs', suffix='electrodes', extension='.tsv',
                         datatype='ieeg', acquisition=bids_path.acquisition,
                         root=bids_path.root)
    elec_df = pd.read_csv(elec_path, delimiter='\t')
    ch_names = elec_df['name'].tolist()
    ch_coords = elec_df[['x', 'y', 'z']].to_numpy(dtype=float) / 1000.
    ch_atlas_labels = elec_df['destriuex'].tolist()

    # transform coordinates to MNI space
    if subject == 'fsaverage':
        tal_trans = mne.read_talxfm(subject=bids_path.subject, subjects_dir=subjects_dir)
        ch_coords = mne.transforms.apply_trans(tal_trans, ch_coords)

    # create channel position coordinate
    ch_pos = dict(zip(ch_names, ch_coords))

    # Ideally the nasion/LPA/RPA will also be present from the digitization, here
    # we use fiducials estimated from the subject's FreeSurfer MNI transformation:
    lpa, nasion, rpa = mne.coreg.get_mni_fiducials(
        subject, subjects_dir=subjects_dir)
    lpa, nasion, rpa = lpa['r'], nasion['r'], rpa['r']
    montage = mne.channels.make_dig_montage(
        ch_pos, coord_frame='mri', nasion=nasion, lpa=lpa, rpa=rpa)
    print('Created %s channel positions' % len(ch_names))

    # create evoked data structure to plot
    info = mne.create_info(ch_names=deriv.ch_names, sfreq=8.0, ch_types='seeg')
    evoked = mne.EvokedArray(deriv.get_data(), info)
    evoked.set_montage(montage)
    trans = mne.channels.compute_native_head_t(montage)
    print(trans)
    print(evoked)
    if show_alignment:
        alignment_figname = figdir / f'sub-{bids_path.subject}_alignment_mni.pdf'
        fig = mne.viz.plot_alignment(evoked.info, trans, 'fsaverage',
                                     subjects_dir=subjects_dir, show_axes=True)

        # fig.savefig(alignment_figname)
        # figure = mlab.gcf()
        screenshot = mlab.screenshot(figure=fig)
        import pylab as pl
        pl.imshow(screenshot)
        pl.axis('off')
        pl.show()
        pl.savefig(alignment_figname)
        # screenshot.save(alignment_figname)
        # mlab.savefig(alignment_figname.as_posix(), figure=fig)

    brain_movie_fname = figdir / f'sub-{bids_path.subject}_mni_fragility_movie.m4'
    brain_fig_snapshot = figdir / f'sub-{bids_path.subject}_mni_fragility.pdf'
    stc_fig_snapshot = figdir / f'sub-{bids_path.subject}_mni_fragility_stc.pdf'

    # get standard fsaverage volume (5mm grid) source space
    fname_src = os.path.join(subjects_dir, 'fsaverage', 'bem',
                             'fsaverage-vol-5-src.fif')
    # fname_src = os.path.join(subjects_dir, subject, 'bem',
    #                          f'{subject}-head.fif')
    vol_src = mne.read_source_spaces(fname_src)

    labels_vol = [
        # 'Left-Amygdala',
        #           'Left-Thalamus-Proper',
        #           'Left-Temporal-Pole',
        # 'ctx_lh_G_front_sup',
        # 'ctx-lh-superiorfrontal',
                  # 'Left-Cerebellum-Cortex',
                  # 'Brain-Stem',
                  # 'Right-Amygdala',
                  # 'Right-Thalamus-Proper',
                  # 'Right-Cerebellum-Cortex'
                  ]
    # fname_aseg = os.path.join(subjects_dir, subject, 'mri', 'aparc.a2009s+aseg.mgz')
    # vol_src = mne.setup_volume_source_space(
    #     subject, mri=fname_aseg,
    #     subjects_dir=subjects_dir,
    #     single_volume=True,
    #     verbose=True,
    #     # add_interpolator=False,  # just for speed, usually this should be True
    #     # pos=10.0,
    #     # bem=fname_model,
    #     # volume_label=labels_vol,
    #     # volume_label = np.unique(ch_atlas_labels).tolist(),
    # )

    # create Source Time Course
    # evoked = evoked.pick_channels(["L'2", "L'3", "L'4",
    #                                "X'12", "X'13", "X'14"])
    # print(evoked)
    evoked = evoked.crop(tmin=0, tmax=30)
    stc = mne.stc_near_sensors(
        evoked, trans, subject, subjects_dir=subjects_dir, src=vol_src,
        distance=0.008,
        mode='single',
        verbose='error')  # ignore missing electrode warnings
    clim = dict(kind='value', lims=
    # [0, 0.4, 0.8]
    np.percentile(evoked.data, [10, 50, 90])
                )

    # restrict to resected region
    print(stc)
    # stc = stc.in_label('ctx-lh-superiorfrontal', mri=fname_aseg,
    #                    src=vol_src)

    # plot Nutmeg style
    fig = stc.plot(src=vol_src, subject=subject, subjects_dir=subjects_dir, clim=clim,
                   colormap='turbo', colorbar=True)
    fig.savefig(brain_fig_snapshot)
    plt.show()

    fname_aseg = os.path.join(subjects_dir, subject, 'mri', 'aparc.a2009s+aseg.mgz')
    label_names = mne.get_volume_labels_from_aseg(fname_aseg)
    label_tc = stc.extract_label_time_course(fname_aseg, src=vol_src)

    lidx, tidx = np.unravel_index(np.argmax(label_tc), label_tc.shape)
    fig, ax = plt.subplots(1)
    ax.plot(stc.times, label_tc.T, 'k', lw=1., alpha=0.5)

    xy = np.array([stc.times[tidx], label_tc[lidx, tidx]])
    xytext = xy + [0.01, 0.025]
    ax.annotate(
        label_names[lidx], xy, xytext, arrowprops=dict(arrowstyle='->'), color='r')

    # print(lidx.shape, tidx.shape)
    # print(np.argsort(label_tc, axis=0)[1])
    # print(np.argmax(label_tc))
    lidx, tidx = np.unravel_index(np.argsort(label_tc, axis=None)[-1],
                                  label_tc.shape)
    # print(lidx.shape, tidx.shape)
    xy = np.array([stc.times[tidx], label_tc[lidx, tidx]])
    xytext = xy + [0.01, 0.025]
    ax.annotate(
        label_names[lidx], xy, xytext, arrowprops=dict(arrowstyle='->'), color='r')

    ax.set(xlim=stc.times[[0, -1]], xlabel='Time (s)',
           ylabel='Fragility Activation at Source')
    for key in ('right', 'top'):
        ax.spines[key].set_visible(False)
    fig.tight_layout()
    plt.show()
    fig.savefig(stc_fig_snapshot)


    # plot 3D movie and save it
    brain = stc.plot_3d(
        src=vol_src, subjects_dir=subjects_dir, colormap='turbo',
        view_layout='horizontal', views=['axial', 'coronal', 'sagittal'],
        size=(800, 300), show_traces=0.4, clim=clim,
        title=f'{bids_path.subject} {bids_path.task} Fragility',
        add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=8)),
        # volume_options=dict(blending='composite'),
    )

    # save brain movie
    brain.save_movie(time_dilation=1, interpolation='linear', framerate=32,
                     time_viewer=True, filename=brain_movie_fname)


def run_plot_heatmap(deriv_path, figures_path):
    subject = 'fsaverage'
    subjects_dir = '/Users/adam2392/Dropbox/epilepsy_bids/derivatives/freesurfer'
    electrodes_fpath = '/Users/adam2392/Downloads/nihcoords/NIH040_2.csv'
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path + '/subjects'
    deriv = read_derivative_npy(deriv_path)

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

    elec_df = pd.read_csv(electrodes_fpath)
    # ch_names = elec_df['chanName'].tolist()
    ch_names = deriv.ch_names
    ch_coords = elec_df.to_numpy(dtype=float) / 1000.
    # ch_coords = elec_df[['x', 'y', 'z']].to_numpy(dtype=float) / 1000.
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

    info = mne.create_info(ch_names=deriv.ch_names, sfreq=8.0, ch_types='ecog')
    raw = mne.EvokedArray(deriv.get_data(), info)

    # attach montage
    raw.set_montage(montage)
    # fig = raw.plot_sensors(ch_type='ecog')
    # plt.show()
    fig = plot_alignment(raw.info, subject=subject, subjects_dir=subjects_dir,
                         # surfaces=['pial'],
                         trans=trans)
    mne.viz.set_3d_view(fig, 200, 70, focalpoint=[0, -0.005, 0.03])

    xy, im = snapshot_brain_montage(fig, montage, hide_sensors=False)

    # paths to mne datasets - sample ECoG and FreeSurfer subject
    # misc_path = mne.datasets.misc.data_path()
    # # In this tutorial, the electrode coordinates are assumed to be in meters
    # elec_df = pd.read_csv(misc_path + '/ecog/sample_ecog_electrodes.tsv',
    #                       sep='\t', header=0, index_col=None)
    # ch_names = elec_df['name'].tolist()
    # ch_coords = elec_df[['x', 'y', 'z']].to_numpy(dtype=float)
    # ch_pos = dict(zip(ch_names, ch_coords))
    # montage = mne.channels.make_dig_montage(
    #     ch_pos, coord_frame='mri', nasion=nasion, lpa=lpa, rpa=rpa)
    # # first we'll load in the sample dataset
    # raw = mne.io.read_raw_edf(misc_path + '/ecog/sample_ecog.edf')
    #
    # # drop bad channels
    # raw.info['bads'].extend([ch for ch in raw.ch_names if ch not in ch_names])
    # raw.load_data()
    # raw.drop_channels(raw.info['bads'])
    # raw.crop(0, 2)  # just process 2 sec of data for speed
    # raw.set_montage(montage)
    # sample_path = mne.datasets.sample.data_path()
    # subject = 'sample'
    # subjects_dir = sample_path + '/subjects'
    # fig = plot_alignment(raw.info, subject=subject, subjects_dir=subjects_dir,
    #                      surfaces=['pial'], trans=trans, coord_frame='mri')
    # mne.viz.set_3d_view(fig, 200, 70, focalpoint=[0, -0.005, 0.03])
    # xy, im = snapshot_brain_montage(fig, montage)

    print(xy)
    print(im)
    print(mne.sys_info(show_paths=True))
    mlab.show()
    # mlab.savefig('/Users/adam2392/Downloads/pt1_brain_fig.pdf')
    exit(0)
    # Convert from a dictionary to array to plot
    # xy_pts = np.vstack([xy[ch] for ch in raw.info['ch_names']])
    #
    # # create an initialization and animation function
    # # to pass to FuncAnimation
    # def init():
    #     """Create an empty frame."""
    #     return paths,
    #
    # def animate(i, activity):
    #     """Animate the plot."""
    #     paths.set_array(activity[:, i])
    #     return paths,
    # cmap = 'turbo'
    # vmin = 0
    # vmax = 0.7
    # # create the figure and apply the animation of the
    # # gamma frequency band activity
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(im)
    # ax.set_axis_off()
    # paths = ax.scatter(*xy_pts.T, c=np.zeros(len(xy_pts)), s=200,
    #                    cmap=cmap, vmin=vmin, vmax=vmax)
    # fig.colorbar(paths, ax=ax)
    # ax.set_title('Gamma frequency over time (Hilbert transform)',
    #              size='large')
    #
    # # avoid edge artifacts and decimate, showing just a short chunk
    # show_power = raw.get_data()[:, 100:700:2]
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                                fargs=(show_power,),
    #                                frames=show_power.shape[1],
    #                                interval=100, blit=True)
    # plt.show(block=True)
    # plt.waitforbuttonpress()
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
        'la02'
    ]

    # pre, Sz, Extraoperative, post
    task = "interictal"
    acquisition = "seeg"
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
                # run_plot_heatmap(deriv_path=deriv_path, figures_path=figures_path)
                run_plot_brain(deriv_path=deriv_path, figures_path=figures_path, root=root)
                # run plot raw data
                # figures_path = figures_root / Path('originalsampling') / "raw" / reference / f"sub-{subject}"
                # run_plot_raw(bids_path=bids_path, resample_sfreq=None, figures_path=figures_path)
