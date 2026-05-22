"""Plot time history analysis results."""

import os
import pickle
import sys

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
    # archetype = 'brbf_6_ii'
    # hz = '21'
    # dr = 'x'
    # gm = '9'

    # identifier = f'{archetype}::cs::{hz}::{gm}::0.001::{dr}::modal::1.0'
    identifier = 'smrf_9_iv::cs::25::33::0.001::x::modal::1.0'

    path = 'extra/structural_analysis/results/results_10_.sqlite'

    db_handler = DB_Handler(db_path=path)
    identifiers = db_handler.list_identifiers()

    assert identifier in identifiers

    dataframe, _, log_content = db_handler.retrieve_data(identifier)
    status = status_from_log(log_content)
    assert status == 'finished'

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


if __name__ == '__main__':
    main()
