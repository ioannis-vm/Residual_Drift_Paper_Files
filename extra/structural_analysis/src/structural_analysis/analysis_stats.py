"""Check analysis status based on the contents of the log files."""

from __future__ import annotations

import re
from datetime import datetime
from itertools import product

import pandas as pd

from extra.structural_analysis.src.util import (
    check_any_line,
    check_last_line,
    file_exists,
    read_study_param,
)


def status_from_log(logfile) -> str:  # noqa: ANN001
    """Parse a logfile and determine the analysis status."""
    pypath = logfile.replace('log', 'log_python')
    ex = file_exists(logfile)
    if not ex:
        return 'not found'
    if file_exists(pypath):
        if check_any_line(pypath, 'Error'):
            return 'error'
    if check_any_line(logfile, 'Analysis interrupted'):
        return 'interrupted'
    if check_any_line(logfile, 'Analysis failed to converge'):
        return 'failed to converge'
    if check_last_line(logfile, 'Analysis finished'):
        return 'finished'
    return 'running'


def get_logtime(logfile, idx):  # noqa: ANN001, ANN201
    """Parse a logfile and determine the time the analysis started."""
    with open(logfile, encoding='utf-8') as f:  # noqa: PTH123
        lines = f.readlines()
    line = lines[idx]
    date_string = line[:22]
    date_format = '%m/%d/%Y %I:%M:%S %p'
    return datetime.strptime(date_string, date_format)


def get_max_subdiv_reported(logfile):  # noqa: ANN001, ANN201
    """Get the largest reported time step subdivision."""
    with open(logfile, encoding='utf-8') as f:  # noqa: PTH123, FURB101
        log_content = f.read()
    pattern = r'\bnum_subdiv: (\d+)\b'
    matches = re.findall(pattern, log_content)
    num_subdiv_values = [int(match) for match in matches]
    if num_subdiv_values:
        return max(num_subdiv_values)
    return None


if __name__ == '__main__':
    nhz = int(read_study_param('extra/structural_analysis/data/study_vars/m'))
    ngm_cs = int(
        read_study_param('extra/structural_analysis/data/study_vars/ngm_cs')
    )

    atypes = ('scbf',)
    stors = ('9',)
    rcs = ('ii',)
    hzs = [f'{i + 1}' for i in range(nhz)]
    gms = [f'gm{i + 1}' for i in range(ngm_cs)]

    keys = []
    vals: dict[str, list] = {
        'status': [],
        'start_time': [],
        'end_time': [],
        'sub': [],
    }
    for at, st, rc, hz, gm, dr in product(atypes, stors, rcs, hzs, gms, ('x', 'y')):
        key = f'{at}-{st}-{rc}-{hz}-{gm}-{dr}'
        keys.append(key)
        path = (
            f'extra/structural_analysis/results/{at}_{st}_{rc}/'
            f'response_modal/{hz}/{gm}/log_{dr}'
        )
        status = status_from_log(path)
        start_time = get_logtime(path, 0)
        end_time = get_logtime(path, -1)
        sub = get_max_subdiv_reported(path)
        vals['status'].append(status)
        vals['start_time'].append(start_time)
        vals['end_time'].append(end_time)
        vals['sub'].append(sub)

    df = pd.DataFrame(
        vals, index=pd.MultiIndex.from_tuples([x.split('-') for x in keys])
    )

    status_df = df['status']
    status_df.value_counts()
