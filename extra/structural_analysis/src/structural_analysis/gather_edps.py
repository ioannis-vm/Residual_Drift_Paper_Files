"""Extract EDPs from sqlite database and store them in a parquet file."""

import pandas as pd
from tqdm import tqdm

from extra.structural_analysis.src.db import DB_Handler


def main() -> None:  # noqa: D103, RUF100
    db_handler = DB_Handler(
        db_path='extra/structural_analysis/results/edp_results_1.sqlite'
    )
    identifiers = db_handler.list_identifiers()

    # # Remove repeated results
    # repeats = {}
    # for identifier in tqdm(identifiers):
    #     if identifier[-2] == '_':
    #         if identifier[:-2] not in repeats:
    #             repeats[identifier[:-2]] = [identifier]
    #         else:
    #             repeats[identifier[:-2]].append(identifier)

    # for rep in tqdm(repeats):
    #     df, _, _ = db_handler.retrieve_data(rep)
    #     for rrep in repeats[rep]:
    #         ddf, _, _ = db_handler.retrieve_data(rep)
    #         if np.all(ddf == df):
    #             db_handler.delete_record(rrep)
    # # Note: Check for repeated results once again.

    dfs = {}
    for identifier in tqdm(identifiers):
        df, _, _ = db_handler.retrieve_data(identifier)
        dfs[identifier] = df

    merged_df = pd.concat(dfs.values(), keys=[tuple(x.split('::')) for x in dfs])
    # split archetype to system, stories, rc
    merged_df.index = pd.MultiIndex.from_tuples(
        [(*(x[0].split('_')), *x[1:]) for x in merged_df.index],
        names=[
            'system',
            'stories',
            'rc',
            'suite',
            'hz',
            'gm',
            'dt',
            'dir',
            'damping',
            'scaling',
            'edp',
            'loc',
            'dir_num',
        ],
    )

    merged_df.index = merged_df.index.droplevel(
        ['suite', 'dt', 'damping', 'scaling', 'dir_num']
    )

    merged_df = pd.DataFrame(merged_df, columns=['value'])

    # add RSN and scaling info
    df_records = pd.read_csv(
        'extra/structural_analysis/results/site_hazard/'
        'required_records_and_scaling_factors_cs.csv',
        index_col=[0, 1, 2],
    )
    rsns = []
    scaling_factors = []
    for idx in merged_df.index:
        system, stories, rc, hazard_level, gm, _, _, _ = idx
        archetype = f'{system}_{stories}_{rc}'
        rsn = int(df_records.at[(archetype, f'hz_{hazard_level}', 'RSN'), gm])  # noqa: PD008
        scaling = df_records.at[(archetype, f'hz_{hazard_level}', 'SF'), gm]  # noqa: PD008
        rsns.append(rsn)
        scaling_factors.append(scaling)

    merged_df['rsn'] = rsns
    merged_df['scaling_factor'] = scaling_factors
    merged_df.to_parquet('data/edp_results_1.parquet')


if __name__ == '__main__':
    main()
