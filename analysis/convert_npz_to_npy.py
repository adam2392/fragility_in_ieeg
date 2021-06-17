# This file used to convert old computed A matrix data to `.npy` files
# The `.npz` old data was used for Nature Neuroscience publication, while
# the updated `.npy` way of storing data is used to be closer to BIDS compliance.
from pathlib import Path
import numpy as np
import json

from eztrack.io import DerivativeArray, create_deriv_info

def main():
    data_dir = Path('/home/adam2392/hdd2/fragility')
    reference = 'average'

    # get all the subjects programmatically
    subjects = [f.name for f in (data_dir / reference).glob('*') if f.is_dir()]

    # loop over all subjects
    for subject in subjects:
        subj_dir = data_dir / reference / subject
        npz_files = subj_dir.glob('*.npz')

        for fpath in npz_files:
            json_fpath = fpath.as_posix().replace('.npz', '.json')

            # load metadata
            with open(json_fpath, 'r') as fin:
                meta_dict = json.load(fin)

            # load actual computation data
            result_dict = dict(np.load(fpath, allow_pickle=True))
            A_mats = result_dict['adjmats']

            # create Derivative structure from eztrack

            print(result_dict.keys())
            print(meta_dict.keys())
            break

        break

if __name__ == '__main__':
    main()