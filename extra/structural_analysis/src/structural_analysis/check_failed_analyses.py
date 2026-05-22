"""Plot time history analysis results."""

import os
import pickle
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from extra.structural_analysis.src.db import DB_Handler


def status_from_log(logfile: str) -> str:
    """
    Parse a logfile and determine the analysis status.

    Parameters
    ----------
    logfile: str
      Path to a log file.

    Returns
    -------
    str
      Analysis status.

    """
    if 'Error' in logfile:
        return 'error'
    if 'Analysis interrupted' in logfile:
        return 'interrupted'
    if 'Analysis failed to converge' in logfile:
        return 'failed to converge'
    if 'Analysis finished' in logfile:
        return 'finished'
    return 'unknown'


def main() -> None:  # noqa: D103, RUF100
    path = 'extra/structural_analysis/results/failed/results_6.sqlite'
    edp_path = 'extra/structural_analysis/results/failed/edp_results_6.sqlite'
    path_c = 'extra/structural_analysis/results/results_1.sqlite'
    edp_path_c = 'extra/structural_analysis/results/edp_results_1.sqlite'

    db_handler = DB_Handler(db_path=path)
    db_handler_edp = DB_Handler(db_path=edp_path)
    db_handler_c = DB_Handler(db_path=path_c)
    db_handler_edp_c = DB_Handler(db_path=edp_path_c)
    identifiers = db_handler.list_identifiers()
    pprint(identifiers)

    identifier = 'smrf_9_iv::cs::27::35::0.001::y::modal::1.0'

    assert identifier in identifiers
    dataframe, metadata, log_content = db_handler.retrieve_data(identifier)
    dataframe_edp, metadata_edp, log_content_edp = db_handler_edp.retrieve_data(
        identifier
    )
    status = status_from_log(log_content)
    print(status)

    # plot
    num_stories = len(dataframe['ID'].columns)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 3))
    for i in range(num_stories):
        ax[1].plot(dataframe['ID'].iloc[:, i])
    ax[1].grid()
    ax[1].set(xlabel='Time (s)', ylabel='Drift', title='Drift time-history')
    ax[0].plot(dataframe['FV', '0'])
    ax[0].set(ylabel='Velocity (in/s2)', title='Ground velocity time-history')
    fig.tight_layout()
    fig.show()

    # update log > can accept
    message = '08/29/2024 02:38:00 PM MANUAL EDIT: results can be accepted as final.'
    log_content += message
    db_handler.delete_record(identifier)
    db_handler_edp.delete_record(identifier)
    db_handler_c.store_data(identifier, dataframe, metadata, log_content)
    db_handler_edp_c.store_data(
        identifier, dataframe_edp, metadata_edp, log_content_edp
    )

    # list of stuff to repeat

    # check duplicates
    duplicate_id = identifier + '_1'
    dataframe, metadata, log_content = db_handler.retrieve_data(identifier)
    dataframe_d, metadata_d, log_content_d = db_handler.retrieve_data(duplicate_id)
    pd.testing.assert_frame_equal(dataframe, dataframe_d)
    assert metadata == metadata_d
    assert log_content == log_content_d

    num_stories = len(dataframe['ID'].columns)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 3))
    for i in range(num_stories):
        ax[1].plot(dataframe['ID'].iloc[:, i], color='blue')
        ax[1].plot(dataframe_d['ID'].iloc[:, i], color='red')
    ax[1].grid()
    ax[1].set(xlabel='Time (s)', ylabel='Drift', title='Drift time-history')
    ax[0].plot(dataframe['FV', '0'], color='blue')
    ax[0].plot(dataframe_d['FV', '0'], color='red', linestyle='dashed')
    ax[0].set(ylabel='Velocity (in/s2)', title='Ground velocity time-history')
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
