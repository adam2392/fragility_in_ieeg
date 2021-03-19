# -*- coding: utf-8 -*-

import re
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from mne.utils import logger


def _expand_channels(ch_list):
    ch_list = [a.replace("’", "'").replace("\n", "").replace(" ", "") for a in ch_list]

    new_list = []
    for string in ch_list:
        if string == "nan":
            continue

        if not string.strip():
            continue

        # A'1,2,5,7
        match = re.match("^([A-Za-z]+[']*)([0-9,]*)([A-Za-z]*)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            numbers = fst_idx.split(",")
            new_list.extend([name + str(char) for char in numbers if char != ","])
            continue

        # A'1
        match = re.match("^([A-Za-z]+[']*)([0-9]+)$", string)
        if match:
            new_list.append(string)
            continue

        # A'1-10
        match = re.match("^([A-Za-z]+[']*)([0-9]+)-([0-9]+)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            new_list.extend(
                [name + str(i) for i in range(int(fst_idx), int(last_idx) + 1)]
            )
            continue

        # A'1-A10
        match = re.match("^([A-Za-z]+[']*)([0-9]+)-([A-Za-z]+[']*)([0-9]+)$", string)
        if match:
            name1, fst_idx, name2, last_idx = match.groups()
            if name1 == name2:
                new_list.extend(
                    [name1 + str(i) for i in range(int(fst_idx), int(last_idx) + 1)]
                )
                continue

        # A'1,B'1,
        match = re.match("^([A-Za-z]+[']*)([0-9,])([A-Za-z]*)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            numbers = fst_idx.split(",")
            new_list.extend([name + str(char) for char in numbers if char != ","])
            continue

        match = string.split(",")
        if match:
            new_list.extend([ch for ch in match])
            continue
        print("expand_channels: Cannot parse this: %s" % string)
    return new_list


class ClinicalContactColumns(Enum):
    """Clinical excel sheet columns to support regular exp expansion."""

    BAD_CONTACTS = "BAD_CONTACTS"
    WM_CONTACTS = "WM_CONTACTS"
    OUT_CONTACTS = "OUT_CONTACTS"
    SOZ_CONTACTS = "SOZ_CONTACTS"
    SPREAD_CONTACTS = "SPREAD_CONTACTS"
    ABLATED_CONTACTS = "ABLATED_CONTACTS"
    RESECTED_CONTACTS = "RESECTED_CONTACTS"


class ClinicalColumnns(Enum):
    """Clinical excel sheet columns to be used."""

    CURRENT_AGE = "SURGERY_AGE"
    ONSET_AGE = "ONSET_AGE"
    ENGEL_SCORE = "ENGEL_SCORE"
    ILAE_SCORE = "ILAE_SCORE"
    OUTCOME = "OUTCOME"
    GENDER = "GENDER"
    HANDEDNESS = "HAND"
    SUBJECT_ID = "PATIENT_ID"
    CLINICAL_COMPLEXITY = "CLINICAL_COMPLEXITY"
    DATE_FOLLOW_UP = "DATE_LAST_FOLLOW_UP"
    ETHNICITY = "ETHNICITY"
    YEARS_FOLLOW_UP = "YEARS_FOLLOW_UP"
    SITE = "CLINICAL_CENTER"


def _filter_column_name(name):
    """Hardcoded filtering of column names."""
    # strip parentheses
    name = name.split("(")[0]
    name = name.split(")")[0]

    # strip whitespace
    name = name.strip()

    return name


def _format_col_headers(df):
    """Hardcoded format of column headers."""
    df = df.apply(lambda x: x.astype(str).str.upper())
    # map all empty to nans
    df = df.fillna(np.nan)
    df = df.replace("NAN", "", regex=True)
    df = df.replace("", "n/a", regex=True)
    return df


def _expand_ch_annotations(df, cols_to_expand):
    """Regular expression expansion of channels."""
    # do some string processing to expand out contacts
    for col in cols_to_expand:
        if col not in df.columns:
            print(f"{col} is not in the DataFrame columns.")
            continue
        # strip out blank spacing
        df[col] = df[col].str.strip()
        # split contacts by ";", ":", or ","
        df[col] = df[col].str.split("; |: |,")
        df[col] = df[col].map(lambda x: [y.strip() for y in x])
        df[col] = df[col].map(lambda x: [y.replace(" ", "-") for y in x])

        # expand channel labels
        df[col] = df[col].apply(lambda x: _expand_channels(x))
    return df


def read_clinical_excel(
        excel_fpath: Union[str, Path],
        subject: str = None,
        keep_as_df: bool = False,
        verbose: bool = False,
):
    """Read clinical datasheet Excel file.

    Turns the entire datasheet into upper-case, and removes any spaces, and parentheses
    in the column names.

    Subjects:
        Assumes that there are rows stratified by subject ID, and columns can be arbitrarily
        added.

    Channels:
        Assumes that there are some columns that are specifically named in the Excel file,
        which we can use regex to expand into a full list of channel names as strings.

    Parameters
    ----------
    excel_fpath : str | pathlib.Path
        The file path for the Excel datasheet.
    subject : str | optional
        The subject that can be used to filter the Excel DataFrame into just a single row
        of that specific subject.
    keep_as_df : bool
        Whether or not to keep the loaded Excel sheet as a DataFrame, or a structured
        dictionary.
    verbose : bool

    Returns
    -------
    df : Dict | pd.DataFrame
    """
    # load in excel file
    df = pd.read_excel(excel_fpath, engine="openpyxl")

    # expand contact named columns
    # lower-case column names
    df.rename(str.upper, axis="columns", inplace=True)
    # filter column names
    column_names = df.columns
    column_mapper = {name: _filter_column_name(name) for name in column_names}
    # remove any markers (extra on clinical excel sheet)
    df.rename(columns=column_mapper, errors="raise", inplace=True)

    # format column headers
    df = _format_col_headers(df)
    # expand channel annotations
    cols_to_expand = [i.value for i in ClinicalContactColumns]
    df = _expand_ch_annotations(df, cols_to_expand=cols_to_expand)

    # remove dataframes that are still Unnamed
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if verbose:
        print("Dataframe that was created looks like: ")
        try:
            print(df.to_markdown())
        except Exception as e:
            print(df.head(2))
            # print(e)

    # if specific subject, then read in that row
    if subject is not None:
        subject = subject.upper()
        if subject not in df[ClinicalColumnns.SUBJECT_ID.value].tolist():
            logger.error(f"Subject {subject} not in Clinical data sheet.")
            return None

        if keep_as_df:
            return df.loc[df[ClinicalColumnns.SUBJECT_ID.value] == subject]
        else:
            return df.loc[df[ClinicalColumnns.SUBJECT_ID.value] == subject].to_dict(
                "records"
            )[0]
    if keep_as_df:
        return df
    else:
        return df.to_dict("dict")
