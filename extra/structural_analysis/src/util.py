"""Utility functions."""

import os
from io import StringIO

import numpy as np
import pandas as pd
from glob2 import glob
from scipy.interpolate import interp1d

GM_DATA_CACHE = {}
SPECTRUM_CACHE = {}


def retrieve_peer_gm_data(rsn, out_type='filenames'):  # noqa: ANN001, ANN201, C901
    """
    Searches all available `_SearchResults.csv` for a given RSN,
    identifies the right folder and retrieves the unscaled RotD50
    response spectrum or the ground motion filenames.

    """  # noqa: D205, D401, RUF100

    def get_gm_data(search_results_file):  # noqa: ANN001, ANN202
        """
        Helper function that reads certain contents from a PEER
        `_SearchResults.csv` file.

        Returns
        -------
        pd.DataFrame
          Data from the ground motion database flatfile.

        """  # noqa: D205, D401, RUF100
        if search_results_file in GM_DATA_CACHE:
            return GM_DATA_CACHE[search_results_file]

        with open(search_results_file, encoding='utf-8') as f:  # noqa: PTH123, FURB101
            contents = f.read()

        contents = contents.split(' -- Summary of Metadata of Selected Records --')[
            1
        ].split('\n\n')[0]
        data = StringIO(contents)

        dataframe = pd.read_csv(data, index_col=2)
        GM_DATA_CACHE[search_results_file] = dataframe

        return dataframe

    # Find all `_SearchResults.csv` files
    files = glob(
        'extra/structural_analysis/data/ground_motions/*/*/_SearchResults.csv'
    )

    # Identify the one containing the specified `rsn`
    identified_file = None
    for file_path in files:
        data = get_gm_data(file_path)
        if rsn in data.index:
            identified_file = file_path
            break

    if not identified_file:
        msg = f'rsn not found: {rsn}'
        raise ValueError(msg)

    rootdir = os.path.dirname(identified_file)  # noqa: PTH120

    if out_type == 'filenames':
        df = get_gm_data(identified_file)

        filenames = df.loc[
            rsn,
            [
                ' Horizontal-1 Acc. Filename',
                ' Horizontal-2 Acc. Filename',
                ' Vertical Acc. Filename',
            ],
        ].to_list()

        result = []
        for filename in filenames:
            if '---' in filename:
                result.append(None)
            else:
                result.append(f'{rootdir}/' + filename.strip())

        return result

    if out_type == 'spectrum':
        if rsn in SPECTRUM_CACHE:
            return SPECTRUM_CACHE[rsn]

        with open(identified_file, encoding='utf-8') as f:  # noqa: PTH123, FURB101
            contents = f.read()

        contents = contents.split(' -- Scaled Spectra used in Search & Scaling --')[
            1
        ].split('\n\n')[0]
        data = StringIO(contents)

        df = pd.read_csv(data, index_col=0)
        # drop stats columns
        df = df.drop(
            columns=[
                'Arithmetic Mean pSa (g)',
                'Arithmetic Mean + Sigma pSa (g)',
                'Arithmetic Mean - Sigma pSa (g)',
            ]
        )
        df.columns = [x.split(' ')[0].split('-')[1] for x in df.columns]
        df.columns.name = 'RSN'
        df.columns = df.columns.astype(int)
        df.index.name = 'T'

        SPECTRUM_CACHE[rsn] = df[rsn]

        return df[rsn]

    if out_type == 'other':
        df = get_gm_data(identified_file)
        return df.loc[rsn]

    msg = f'Unsupported out_type: {out_type}'
    raise ValueError(msg)


def retrieve_peer_gm_spectra(rsns):  # noqa: ANN001, ANN201
    """
    Uses retrieve_peer_gm_data to prepare a dataframe with response
    spectra for the given RSNs.
    """  # noqa: D205, D401, RUF100
    rsn_dfs = []
    for rsn in rsns:
        rsn_df = retrieve_peer_gm_data(rsn, out_type='spectrum')
        rsn_dfs.append(rsn_df)
    return pd.concat(rsn_dfs, keys=rsns, axis=1)


def interpolate_pd_series(series, values):  # noqa: ANN001, ANN201
    """Interpolates a pandas series for specified index values."""
    idx_vec = series.index.to_numpy()
    vals_vec = series.to_numpy()
    ifun = interp1d(idx_vec, vals_vec, kind='linear', fill_value='extrapolate')
    if isinstance(values, float):
        return float(ifun(values))
    if isinstance(values, np.ndarray):
        return ifun(values)
    return ValueError(f'Invalid datatype: {type(values)}')


def file_exists(file_path):  # noqa: ANN001, ANN201
    """
    Checks if a file exists at the specified file path.

    Args:
        file_path (str): The path to the file.

    Returns
    -------
        bool: True if the file exists, False otherwise.
    """  # noqa: D401, RUF100
    return os.path.exists(file_path) and os.path.isfile(file_path)  # noqa: PTH110, PTH113


def check_last_line(file_path, target_string) -> bool:  # noqa: ANN001
    """
    Checks if the last line of a file contains a specific string.

    Args:
        file_path (str): The path to the file.
        target_string (str): The string to search for in the last line.

    Returns
    -------
        bool: True if the last line contains the target string, False otherwise.
    """  # noqa: D401, RUF100
    with open(file_path, encoding='utf-8') as file:  # noqa: PTH123
        lines = file.readlines()

    # Check if the file is not empty
    if lines:
        last_line = lines[-1].strip()  # Remove leading/trailing whitespace

        # Check if the last line contains the target string
        if target_string in last_line:
            return True

    return False


def check_any_line(file_path, target_string) -> bool:  # noqa: ANN001
    """
    Checks if any line of a file contains a specific string.

    Args:
        file_path (str): The path to the file.
        target_string (str): The string to search for in the last line.

    Returns
    -------
        bool: True if the last line contains the target string, False otherwise.
    """  # noqa: D401, RUF100
    with open(file_path, encoding='utf-8') as file:  # noqa: PTH123, FURB101
        all_contents = file.read()

    # Check if the file is not empty
    if all_contents:
        if target_string in all_contents:
            return True

    return False


def get_any_line(file_path, target_string):  # noqa: ANN001, ANN201
    """
    Checks if any line of a file contains a specific string.
    If it does, it returns that line.

    Args:
        file_path (str): The path to the file.
        target_string (str): The string to search for in the last line.

    Returns
    -------
        str: The line
    """  # noqa: D205, D401, RUF100
    with open(file_path, encoding='utf-8') as file:  # noqa: PTH123
        all_contents = file.readlines()

    # Check if the file is not empty
    if all_contents:
        for line in all_contents:
            if target_string in line:
                return line

    return None


def check_logs(path) -> str:  # noqa: ANN001
    """Check the logs of a nonlinear analysis."""
    exists = os.path.exists(path) and os.path.isfile(path)  # noqa: PTH110, PTH113
    if not exists:
        return 'does not exist'
    inter = check_any_line(path, 'Analysis interrupted')
    if inter:
        return 'interrupted'
    fail = check_any_line(path, 'Analysis failed to converge')
    if fail:
        return 'failed'
    return 'finished'


def read_study_param(param_path):  # noqa: ANN001, ANN201
    """Read a study parameter from a file."""
    with open(param_path, encoding='utf-8') as f:  # noqa: PTH123, FURB101
        return f.read()
