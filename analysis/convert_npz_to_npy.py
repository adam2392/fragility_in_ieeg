# This file used to convert old computed A matrix data to `.npy` files
# The `.npz` old data was used for Nature Neuroscience publication, while
# the updated `.npy` way of storing data is used to be closer to BIDS compliance.
from pathlib import Path
import numpy as np
import json

from eztrack.io import DerivativeArray, create_deriv_info
from eztrack.io.base import DERIVATIVETYPES

def main():
    data_dir = Path('/home/adam2392/hdd2/fragility')
    deriv_root = Path('/home/adam2392/hdd2/ictalfragility/')
    reference = 'average'

    root = Path('/home/adam2392/hdd2/epilepsy_bids')

    overwrite = True

    # get all the subjects programmatically
    subjects = [f.name for f in (data_dir / reference).glob('*') if f.is_dir()]

    # loop over all subjects
    for subject in subjects:
        subj_dir = data_dir / reference / subject
        npz_files = subj_dir.glob('*.npz')

        deriv_chain = Path("fragility") / reference / f"sub-{subject}"
        deriv_path = deriv_root / deriv_chain

        for fpath in npz_files:
            json_fpath = fpath.as_posix().replace('.npz', '.json')

            # load metadata
            with open(json_fpath, 'r') as fin:
                meta_dict = json.load(fin)

            # load actual computation data
            result_dict = dict(np.load(fpath, allow_pickle=True))
            A_mats = result_dict['adjmats']
            pert_mats = result_dict['pertmats']
            delta_vecs_arr = result_dict['delta_vecs_mat']

            ch_names = meta_dict['ch_names']
            sfreq = meta_dict['sfreq']

            # set parameters
            order = meta_dict.get("order", 1)
            winsize = meta_dict.get("winsize", 250 * order)
            stepsize = meta_dict.get("stepsize", 125 * order)
            radius = meta_dict.get("radius", 1.5)
            perturb_type = meta_dict.get("perturb_type", "C")
            method_to_use = meta_dict.get("method_to_use", "pinv")
            l2penalty = meta_dict.get("l2penalty", 0)
            weighted = meta_dict.get("weighted", False)

            deriv_sfreq = sfreq / stepsize
            
            ltvmodel_kwargs = {
                "l2penalty": l2penalty,
                "order": order,
                "method_to_use": method_to_use,
                "solver": "auto",
                "fit_intercept": True,
                "normalize": True,
                "weighted": weighted,
            }
            ltvmodel_kwargs.update(
                {
                    "winsize": winsize,
                    "stepsize": stepsize,
                }
            )

            pertmodel_kwargs = {
                "radius": radius,
                "perturb_type": perturb_type,
                # "on_error": model_params.get("on_error", "ignore"),
            }
            # create Derivative structure from eztrack
            deriv_info = create_deriv_info(
                ch_names=ch_names,
                sfreq=deriv_sfreq,
                description=DERIVATIVETYPES.STATE_MATRIX.value,
                # rawsources=raw.filenames,
                # sources=raw.filenames,
                reference=reference,
                model_parameters=ltvmodel_kwargs,
                ch_axis=[0, 1],
                source_info=None,
            )

            state_arr_deriv = DerivativeArray(
                A_mats, info=deriv_info, first_samp=0, copy="auto")
            
            # create DerivativeInfo and Array data structures
            # for the perturbation norm array
            perturb_type = perturb_type.upper()
            if perturb_type == "C":
                description = DERIVATIVETYPES.COLPERTURB_MATRIX.value
                delta_description = DERIVATIVETYPES.DELTAVECS_MATRIX.value
            elif perturb_type == "R":
                description = DERIVATIVETYPES.ROWPERTURB_MATRIX.value
                delta_description = DERIVATIVETYPES.ROWDELTAVECS_MATRIX.value

            # create min norm delta vectors info
            deltavecs_info = create_deriv_info(
                ch_names=ch_names,
                sfreq=deriv_sfreq,
                description=delta_description,
                reference=reference,
                model_parameters=pertmodel_kwargs,
                ch_axis=[0],
            )
            delta_vecs_arr_deriv = DerivativeArray(
                delta_vecs_arr, info=deltavecs_info, first_samp=0, copy="auto"
            )

            # create min norm perturbation info
            deriv_info = create_deriv_info(
                ch_names=ch_names,
                sfreq=deriv_sfreq,
                description=description,
                reference=reference,
                model_parameters=pertmodel_kwargs,
            )
            perturb_deriv = DerivativeArray(
                pert_mats, info=deriv_info, first_samp=0, copy="auto"
            )

            perturb_deriv_fpath = deriv_path / perturb_deriv.info._expected_basename
            state_deriv_fpath = deriv_path / state_arr_deriv.info._expected_basename
            delta_vecs_deriv_fpath = deriv_path / delta_vecs_arr_deriv.info._expected_basename

            print("Saving files to: ")
            print(perturb_deriv_fpath)
            print(state_deriv_fpath)
            print(delta_vecs_deriv_fpath)
            perturb_deriv.save(perturb_deriv_fpath, overwrite=overwrite)
            state_arr_deriv.save(state_deriv_fpath, overwrite=overwrite)
            delta_vecs_arr_deriv.save(delta_vecs_deriv_fpath, overwrite=overwrite)

            print(result_dict.keys())
            print(meta_dict.keys())
            break

        break

if __name__ == '__main__':
    main()