import os
import pandas as pd

def read_participants_tsv(root, participant_id=None, column_keys=None):
    """Read in ``participants.tsv`` file.

    Allows one to query the columns of the file as
    a DataFrame and say obtain the list of Engel scores.
    """
    participants_tsv_fname = os.path.join(root, 'participants.tsv')

    participants_df = pd.read_csv(participants_tsv_fname, delimiter='\t')
    if participant_id is not None:
        participants_df = participants_df[participants_df['participant_id'] == participant_id]

    if column_keys is not None:
        if not isinstance(column_keys, list):
            column_keys = [column_keys]
        participants_df = participants_df[column_keys]
    return participants_df

