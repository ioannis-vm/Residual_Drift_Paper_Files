"""
Read time-history analysis results form databases and extract the
relevant EDPs.

Revised on Wed Aug 28 01:10:10 PM PDT 2024
to add signed EDPs and ground motion displacement.

"""

from __future__ import annotations

import importlib
import logging
import os
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI
from osmg import solver
from osmg.ground_motion_utils import import_PEER
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

from extra.structural_analysis.src.db import DB_Handler
from extra.structural_analysis.src.util import retrieve_peer_gm_data


def status_from_log(logfile: str) -> str:
    """
    Parse a logfile and determine the analysis status.

    Returns
    -------
    str
      The status.

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


def obtain_edps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the EDPs from a dataframe containing the full
    time-history analysis results.
    The EDPs are `PFA`, `PFV`, `PID`, and `RID`.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe containing the full time-history analysis results.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the EDPs.

    """
    # get signed maximum values. That is, if the largest value in
    # absolute terms is negative, keep the negative sign:
    dataframe = dataframe.reset_index().drop(
        ['Rtime', 'Subdiv', 'Vb', 'time'], axis=1, level=0
    )
    edps = pd.Series(index=dataframe.columns)
    for col in dataframe.columns:
        edps.loc[col] = dataframe[col].loc[dataframe[col].abs().idxmax()]
    edps['FA'] /= 386.22
    edps.index = pd.MultiIndex.from_tuples(
        [(f'P{x[0]}', x[1], x[2]) for x in edps.index]
    )
    rid = dataframe.iloc[-1, :]['ID']
    rid.index = pd.MultiIndex.from_tuples([('RID', x[0], x[1]) for x in rid.index])
    res = pd.concat((edps, rid), axis=0)
    res.index.names = ['edp', 'loc', 'dr']
    return res


def parse_identifier(identifier: str) -> dict[str, str]:
    """
    Parse an identifier to extract analysis information.

    Parameters
    ----------
    identifier: str
      The identifier to parse

    Returns
    -------
    dict
      Dictionary with analysis information.

    """
    parts = identifier.split('::')
    archetype = parts[0]
    system = archetype.split('_')[0]
    stories = archetype.split('_')[1]
    rc = archetype.split('_')[2]
    hz = parts[2]
    gm = parts[3]
    dr = parts[5]
    return {
        'system': system,
        'stories': stories,
        'rc': rc,
        'hz': hz,
        'gm': gm,
        'dr': dr,
    }


def get_gm_data(rsn: int, scaling: float, dr: str, t1: float) -> tuple[float]:
    """
    Get ground motion-related EDP data.

    Parameters
    ----------
    rsn: int
      Record sequence number.
    scaling: float
      Scaling factor.
    dr: str
      Direction. Any of {'x', 'y'}.
    t1: float
      First-mode period.

    Returns
    -------
    tuple
      PGA (g), SA(T1) (g), PGV (in/s), PGD (in) , RGD (in)

    """
    dr_idx = {'x': 0, 'y': 1}
    gm_filename = retrieve_peer_gm_data(rsn)[dr_idx[dr]]
    gm_spectrum = retrieve_peer_gm_data(rsn, out_type='spectrum')
    gm_meta = retrieve_peer_gm_data(rsn, out_type='other')
    gm_data = import_PEER(gm_filename)  # acceleration, in [g] units.
    time = gm_data[:, 0]
    acceleration = gm_data[:, 1] * 386.22  # in/s2
    velocity = cumulative_trapezoid(acceleration, time, initial=0.00)  # in/s
    displacement = cumulative_trapezoid(velocity, time, initial=0.00)

    pga = gm_spectrum.iloc[0] * scaling
    sat1 = np.interp(t1, gm_spectrum.index, gm_spectrum.values) * scaling
    pgv = velocity[np.argmax(np.abs(velocity))] * scaling
    pgd = displacement[np.argmax(np.abs(displacement))] * scaling
    rgd = displacement[-1] * scaling
    t_p = gm_meta[' Tp-Pulse Period (sec)']
    t_p = t_p if t_p != ' -' else None
    dur_5_75 = gm_meta[' 5-75% Duration (sec)']
    dur_5_75 = float(dur_5_75) if dur_5_75 != ' -' else None
    dur_5_95 = gm_meta[' 5-95% Duration (sec)']
    dur_5_95 = float(dur_5_95) if dur_5_95 != ' -' else None
    ai = gm_meta[' Arias Intensity (m/sec)']  # Note: this is in [m/s]
    ai = float(ai) * scaling**2 if ai != ' -' else None
    mag = float(gm_meta[' Magnitude'])
    mech = gm_meta[' Mechanism'].strip()
    rrup = float(gm_meta[' Rrup (km)'])  # Note: this is in [km]
    vs30 = float(gm_meta[' Vs30 (m/sec)'])  # Note: this is in [m/s]

    return {
        'RSN': rsn,
        'SF': scaling,
        'PGA': pga,
        'SAT1': sat1,
        'PGV': pgv,
        'PGD': pgd,
        'RGD': rgd,
        'TPulse': t_p,
        'D575': dur_5_75,
        'D595': dur_5_95,
        'AI': ai,
        'MW': mag,
        'MECH': mech,
        'RRUP': rrup,
        'VS30': vs30,
    }


def get_rsn_and_scaling(
    archetype: str, hz: str, gm: str, gm_data: pd.DataFrame
) -> tuple[int, float]:
    """
    Get the RSN and scaling factor of a given analysis.

    Parameters
    ----------
    archetype: str
      Archetype code.
    hz: str
      Hazard level.
    gm: str
      Ground motion number.
    gm_data: pd.DataFrame
      Dataframe containing ground motion RSNs and scaling factors for
      all cases.

    Returns
    -------
    tuple
      RSN and scaling factor.

    """
    if archetype == 'brbf_1_ii':
        archetype = 'brbf_3_ii'
    data = gm_data.loc[archetype, f'hz_{hz}'].loc[:, gm].to_dict()
    return int(data['RSN']), data['SF']


def get_archetype_periods() -> dict[str, float]:
    """
    Get the period of all archetypes.

    Raises
    ------
    ValueError
      If an archetype code is invalid.

    Returns
    -------
    Dictionary containing the first-mode period of each archetype.

    """
    period_dict = {}

    for system, stories, rc in tqdm(
        list(product(('smrf', 'scbf', 'brbf'), ('3', '6', '9'), ('ii', 'iv')))  # noqa: RUF005
        + [('brbf', '1', 'ii')]
    ):
        archetype = f'{system}_{stories}_{rc}'
        archetypes_module = importlib.import_module(
            'extra.structural_analysis.src.structural_analysis.archetypes_2d'
        )
        try:
            archetype_builder = getattr(archetypes_module, archetype)
        except AttributeError as exc:
            msg = f'Invalid archetype code: {archetype}'
            raise ValueError(msg) from exc

        mdl, loadcase = archetype_builder('x')
        modal_analysis = solver.ModalAnalysis(
            mdl, {loadcase.name: loadcase}, num_modes=int(stories) * 6
        )
        modal_analysis.settings.store_forces = False
        modal_analysis.settings.store_fiber = False
        modal_analysis.settings.restrict_dof = [
            False,
            True,
            False,
            True,
            False,
            True,
        ]
        modal_analysis.run()
        periods = modal_analysis.results[loadcase.name].periods
        period_dict[archetype] = periods[0]
    return period_dict


def main() -> None:
    """Main method."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    logging.info('Obtaining the predominant period of each archetype.')  # noqa: LOG015

    # archetype_periods = get_archetype_periods()
    archetype_periods = {
        'brbf_1_ii': 0.2768514650059095,
        'brbf_3_ii': 0.5442384847492099,
        'brbf_3_iv': 0.437180973809568,
        'brbf_6_ii': 0.8136688862189864,
        'brbf_6_iv': 0.612289299643867,
        'brbf_9_ii': 1.1667836439917456,
        'brbf_9_iv': 0.8214565671719624,
        'scbf_3_ii': 0.3580825197264572,
        'scbf_3_iv': 0.26623709819954744,
        'scbf_6_ii': 0.5784823678528747,
        'scbf_6_iv': 0.44342791361369444,
        'scbf_9_ii': 0.8096678364166747,
        'scbf_9_iv': 0.5626766190315841,
        'smrf_3_ii': 0.6906464655699505,
        'smrf_3_iv': 0.4522496981121477,
        'smrf_6_ii': 1.0299703407225254,
        'smrf_6_iv': 0.7025975739930206,
        'smrf_9_ii': 1.3821830258022287,
        'smrf_9_iv': 0.8302257292099686,
    }

    logging.info('Getting EDPs from time-history results.')  # noqa: LOG015

    edp_data = {}

    database_path = Path('extra/structural_analysis/results/results_1.sqlite')

    db_handler = DB_Handler(db_path=database_path)
    identifiers = db_handler.list_identifiers()

    num_identifiers_total = len(identifiers)
    num_per_thread = int(float(num_identifiers_total) / float(size))
    i_start = int(rank) * num_per_thread
    i_end = i_start + num_per_thread
    i_end = min(i_end, num_identifiers_total - 1)
    identifiers = identifiers[i_start:i_end]

    for identifier in tqdm(identifiers):
        dataframe, _, _ = db_handler.retrieve_data(identifier)
        edps = obtain_edps(dataframe)
        edp_data[identifier] = edps

    gm_data = (
        pd.read_csv(
            'extra/structural_analysis/results/site_hazard/'
            'required_records_and_scaling_factors_cs.csv'
        )
        .set_index(['archetype', 'hazard_level', 'quantity'])
        .sort_index()
    )

    logging.info('Getting IM data from ground motions.')  # noqa: LOG015
    gm_im_data = {}
    for identifier in tqdm(identifiers):
        analysis_data = parse_identifier(identifier)
        archetype = (
            f'{analysis_data["system"]}_'
            f'{analysis_data["stories"]}_'
            f'{analysis_data["rc"]}'
        )
        t1 = archetype_periods[archetype]  # TODO(JVM): update this
        rsn, scaling = get_rsn_and_scaling(
            archetype,
            analysis_data['hz'],
            analysis_data['gm'],
            gm_data,
        )
        gm_im_data[identifier] = get_gm_data(rsn, scaling, analysis_data['dr'], t1)

    logging.info('Extending existing dataframes.')  # noqa: LOG015
    # extend the EDP dataframes to include the IM data
    for identifier in tqdm(identifiers):
        identifier_data = parse_identifier(identifier)
        edp_data[identifier] = (
            edp_data[identifier].reset_index().rename({0: 'value'}, axis=1)
        )
        for key, value in gm_im_data[identifier].items():
            edp_data[identifier][key] = value
        for key, value in identifier_data.items():
            edp_data[identifier][key] = value

    edp_dataframe = pd.concat(edp_data.values())
    edp_dataframe = edp_dataframe.set_index(
        [
            'system',
            'stories',
            'rc',
            'hz',
            'gm',
            'dr',
            'loc',
            'edp',
            'RSN',
            'SF',
            'PGA',
            'SAT1',
            'PGV',
            'PGD',
            'RGD',
            'TPulse',
            'D575',
            'D595',
            'AI',
            'MW',
            'MECH',
            'RRUP',
            'VS30',
        ]
    ).unstack('edp')
    edp_dataframe.columns = edp_dataframe.columns.droplevel(0)
    edp_dataframe.columns.name = None
    edp_dataframe = edp_dataframe.reset_index()

    edp_dataframe.to_parquet(
        f'extra/structural_analysis/results/signed_edp_data_{rank}.parquet'
    )

    comm.Barrier()

    if rank == 0:
        files = [  # noqa: C416
            f
            for f in Path('extra/structural_analysis/results/').rglob(
                'signed_edp_data_*.parquet'
            )
        ]

        all_data = []

        for file in tqdm(files):
            data = pd.read_parquet(file)
            all_data.append(data)

        all_data_df = pd.concat(all_data)

        all_data_df = all_data_df.set_index(
            ['system', 'stories', 'rc', 'hz', 'gm', 'dr', 'loc']
        )
        all_data_df = all_data_df.sort_index()
        all_data_df.index.is_monotonic_increasing  # noqa: B018
        all_data_df.index.is_unique  # noqa: B018
        all_data_df = all_data_df.reset_index()

        all_data_df.to_parquet('extra/structural_analysis/results/data.parquet')

        for file in tqdm(files):
            file.unlink()

    # cols = [
    #     'PGA',
    #     'SAT1',
    #     'PGV',
    #     'PGD',
    #     'RGD',
    #     'PFA',
    #     'PFV',
    #     'PID',
    #     'RID',
    #     'TPulse',
    #     'D575',
    #     'D595',
    #     'AI',
    #     'MW',
    #     'RRUP',
    #     'VS30',
    # ]
    # edp_sub = edp_dataframe[cols]
    # corr_df = edp_sub.corr()
    # corr_df[corr_df > 0.20]

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # plt.close()
    # fig, ax = plt.subplots()
    # sns.scatterplot(
    #     x='PID',
    #     y='RID',
    #     data=edp_dataframe[['PID', 'RID']].dropna(how='any').astype(float),
    # )
    # ax.grid(which='both', linewidth=0.30)
    # plt.show()

    # for i, other in enumerate(
    #     [
    #         'PGA',
    #         'SAT1',
    #         'PGV',
    #         'PGD',
    #         'RGD',
    #         'PFA',
    #         'PFV',
    #         'TPulse',
    #         'D575',
    #         'D595',
    #         'AI',
    #         'MW',
    #         'RRUP',
    #         'VS30',
    #     ]
    # ):
    #     plt.close()
    #     fig, ax = plt.subplots(figsize=(3, 3))
    #     sns.scatterplot(
    #         x='PID',
    #         y='RID',
    #         data=edp_dataframe[['PID', 'RID', other]]
    #         .dropna(how='any')
    #         .astype(float),
    #         hue=other,
    #         palette='vlag',
    #     )
    #     ax.grid(which='both', linewidth=0.30)
    #     ax.set(title=other)
    #     fig.tight_layout()
    #     fig.savefig(f'/tmp/fig_{i}.png', dpi=1800)
    #     # plt.show()

    # edp_dataframe['RGD'].abs()[edp_dataframe['RGD'].abs() > 0.10]
    # rsn = edp_dataframe.loc[932]['RSN']

    # gm_filename = retrieve_peer_gm_data(rsn)[0]
    # gm_spectrum = retrieve_peer_gm_data(rsn, out_type='spectrum')
    # gm_meta = retrieve_peer_gm_data(rsn, out_type='other')
    # gm_data = import_PEER(gm_filename)  # acceleration, in [g] units.
    # time = gm_data[:, 0]
    # acceleration = gm_data[:, 1] * 386.22  # in/s2
    # velocity = cumulative_trapezoid(acceleration, time, initial=0.00)  # in/s
    # displacement = cumulative_trapezoid(velocity, time, initial=0.00)

    # fig, ax = plt.subplots(3, 1, sharex=True)
    # ax[0].plot(time, displacement)
    # ax[1].plot(time, velocity)
    # ax[2].plot(time, acceleration)
    # for xx in ax:
    #     xx.grid(which='both', linewidth=0.30)
    # ax[0].set(ylabel='D')
    # ax[1].set(ylabel='V')
    # ax[2].set(ylabel='A')
    # plt.show()

    # fig, ax = plt.subplots(figsize=(3, 3))
    # ddd = edp_dataframe[['RGD', 'PGD']].dropna(how='any').astype(float)
    # ddd['ratio'] = ddd['RGD'] / ddd['PGD']
    # sns.boxplot(x='ratio', data=ddd, ax=ax)
    # ax.set(xlabel='Residual/Peak Ground Displacement (in)')
    # fig.tight_layout()
    # plt.savefig('/tmp/fig.png', dpi=1800)
    # plt.show()


if __name__ == '__main__':
    main()
