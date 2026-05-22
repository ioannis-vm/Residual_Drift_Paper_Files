"""Functions to process the input data."""

import pandas as pd


def load_dataset(
    path: str = 'data/signed_edp_data.parquet',
) -> pd.DataFrame:
    data = pd.read_parquet(path)
    return (
        data.set_index(['system', 'stories', 'rc', 'hz', 'gm', 'dr', 'loc'])
        .dropna(subset='PID')
        .sort_index()
    )


def remove_collapse(
    data: pd.DataFrame, drift_threshold: float = 0.08
) -> pd.DataFrame:
    """
    Remove collapse instances.

    Collapse is determined by cases where the peak drift exceeds a
    threshold in any locaiton or direction.

    Parameters
    ----------
    data: pd.DataFrame
      Dataframe containing PID-RID pairs.

    Returns
    -------
    pd.DataFrame
      Dataframe with collapse instances removed. The returned
      dataframe will only include cases where the peak drift does not
      exceed the specified threshold.

    """
    data_unstack = data.unstack(['loc', 'dr'])
    no_collapse = data_unstack['PID'].abs().max(axis=1) < drift_threshold
    data_unstack_no_collapse = data_unstack.loc[no_collapse, :]
    return data_unstack_no_collapse.stack(
        ['loc', 'dr'], future_stack=True
    ).sort_index()


def only_drifts(data: pd.DataFrame) -> pd.DataFrame:
    return data[['PID', 'RID']].dropna(how='all')


def pick_column(row: pd.Series, pidstory_series: pd.Series) -> object:
    """
    Select a value from the given row.

    Select a value from the given row based on the corresponding value
    in the PIDStory series.

    Parameters
    ----------
    row: pd.Series
        A row from the `pid_rid` DataFrame.
    pidstory_series : pd.Series
        A series containing the PIDStory values, with indices matching
        the `pid_rid` DataFrame.

    Returns
    -------
    value: scalar
        The value from the `row` that corresponds to the PIDStory
        value for that row's index.

    """
    pidstory_value = pidstory_series.loc[row.name]
    return row[str(pidstory_value)]


def retrieve_values(
    system: str,
    stories: str,
    rc: str,
    story: str,
    data: pd.DataFrame,
    *,
    apply_abs: bool = False,
) -> pd.DataFrame:
    """
    Get PID-RID pairs and other metadata.

    Parameters
    ----------
    system: str
      Structural system.
    stories: str
      Number of stories.
    rc: str
      Risk category.
    story: str
      Story to use. Can be a single story or `max-max`, meaning the
      maximum PID of all stories combined with the corresponding RID
      of the same story (even though some other story has a higher
      RID).
    data: pd.DataFrame
      Results dataset.
    apply_abs:
      If set to True, it returns the absolute value of the PID and RID
      columns.

    Returns
    -------
    pd.DataFrame
      Dataframe with the appropriate subset of the dataset.

    """
    data = data.loc[system, stories, rc].dropna(how='all').sort_index()

    if story == 'max-max':
        data = data.unstack(['loc'])

        data['PIDStory'] = data['PID'].abs().idxmax(axis=1).astype(int)
        data['RIDStory'] = data['RID'].abs().idxmax(axis=1).astype(int)

        pid_columns = data['PID']
        rid_columns = data['RID']
        data = data.drop(['PID', 'RID'], axis=1)
        pidstory_series = data['PIDStory']

        pid_column = pid_columns.apply(
            lambda row: pick_column(row, pidstory_series), axis=1
        )
        rid_column = rid_columns.apply(
            lambda row: pick_column(row, pidstory_series), axis=1
        )
        data['PID'] = pid_column
        data['RID'] = rid_column

    else:
        data.index = data.index.reorder_levels(['loc', 'hz', 'gm', 'dr'])
        data = data.loc[story, :]

    if apply_abs:
        data['PID'] = data['PID'].abs()
        data['RID'] = data['RID'].abs()

    return data
