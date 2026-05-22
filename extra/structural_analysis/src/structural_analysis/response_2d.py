"""Run nonlinear time-history analysis to get the building's response."""

import argparse
import importlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from osmg import solver
from osmg.gen.query import ElmQuery
from osmg.ground_motion_utils import import_PEER

from extra.structural_analysis.src.db import DB_Handler
from extra.structural_analysis.src.util import retrieve_peer_gm_data
from src.util import store_info


def obtain_edps(dataframe):  # noqa: ANN001, ANN201
    """
    Extracts the EDPs from a dataframe containing the full
    time-history analysis results.
    The EDPs are `PFA`, `PFV`, `PID`, and `RID`.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe containing the full time-hisotyr analysis results.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the EDPs

    """  # noqa: D205, D401, RUF100
    edps = dataframe.abs().max().drop(['Rtime', 'Subdiv', 'Vb'])
    edps['FA'] /= 386.22
    edps.index = pd.MultiIndex.from_tuples(
        [(f'P{x[0]}', x[1], x[2]) for x in edps.index]
    )
    rid = dataframe.iloc[-1, :]['ID'].abs()
    rid.index = pd.MultiIndex.from_tuples([('RID', x[0], x[1]) for x in rid.index])
    return pd.concat((edps, rid), axis=0)


def main() -> None:  # noqa: C901, D103, PLR0914, PLR0915, RUF100
    # ~~~~~~~~~~~~~~~~~~~~~~ #
    # set up argument parser #
    # ~~~~~~~~~~~~~~~~~~~~~~ #

    # import sys
    # sys.argv = [
    #     "python",
    #     "--archetype",
    #     "brbf_6_ii",
    #     "--suite_type",
    #     "cs",
    #     "--hazard_level",
    #     "29",
    #     "--gm_number",
    #     "1",
    #     "--analysis_dt",
    #     "0.001",
    #     "--direction",
    #     "x",
    #     '--damping',
    #     "modal",
    #     '--scaling',
    #     "1.00",
    #     '--group_id',
    #     '99999',
    # ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--archetype')
    parser.add_argument('--suite_type')
    parser.add_argument('--hazard_level')
    parser.add_argument('--gm_number')
    parser.add_argument('--analysis_dt')
    parser.add_argument('--direction')
    parser.add_argument('--damping')
    parser.add_argument('--scaling')
    parser.add_argument('--group_id')
    parser.add_argument('--no_LLRS', action='store_true')
    parser.add_argument('--pinned_beams', action='store_true')

    args = parser.parse_args()
    archetype = args.archetype
    suite_type = args.suite_type
    hazard_level = args.hazard_level
    gm_number = int(args.gm_number)
    analysis_dt = float(args.analysis_dt)
    direction = args.direction
    damping = args.damping
    additional_scaling = float(args.scaling)
    no_llrs = args.no_LLRS
    pinned_beams = args.pinned_beams

    def split_archetype(archetype):  # noqa: ANN001, ANN202
        system, stories, rc = archetype.split('_')
        stories = int(stories)
        return system, stories, rc

    # load archetype building
    archetypes_module = importlib.import_module(
        'extra.structural_analysis.src.structural_analysis.archetypes_2d'
    )
    try:
        archetype_builder = getattr(archetypes_module, archetype)
    except AttributeError as exc:
        msg = f'Invalid archetype code: {archetype}'
        raise ValueError(msg) from exc

    system, stories, rc = split_archetype(archetype)
    if system != 'smrf':
        mdl, loadcase = archetype_builder(
            direction, no_llrs=no_llrs, pinned_beams=pinned_beams
        )
    else:
        mdl, loadcase = archetype_builder(direction, no_llrs=no_llrs)

    num_levels = len(mdl.levels) - 1
    level_heights = np.diff([level.elevation for level in mdl.levels.values()])

    lvl_nodes = []
    base_node = next(iter(mdl.levels[0].nodes.values())).uid
    lvl_nodes.append(base_node)

    for i in range(num_levels):
        lvl_nodes.append(loadcase.parent_nodes[i + 1].uid)  # noqa: PERF401

    specific_nodes = lvl_nodes + [n.uid for n in mdl.levels[0].nodes.values()]
    # also add the leaning column nodes due to their rotational restraints
    eqr = ElmQuery(mdl)
    for lvl in range(num_levels):
        level_node = eqr.search_node_lvl(0.00, 0.00, lvl + 1)
        assert level_node is not None
        specific_nodes.append(level_node.uid)

    if suite_type == 'cs':
        df_records = pd.read_csv(
            'extra/structural_analysis/results/site_hazard/'
            'required_records_and_scaling_factors_cs.csv',
            index_col=[0, 1, 2],
        )

        rsn = int(
            df_records.at[  # noqa: PD008
                (archetype, f'hz_{hazard_level}', 'RSN'), str(gm_number)
            ]  # noqa: PD008, RUF100
        )
        scaling = df_records.at[  # noqa: PD008
            (archetype, f'hz_{hazard_level}', 'SF'), str(gm_number)
        ]

        dir_idx = {'x': 0, 'y': 1}
        try:
            gm_filename = retrieve_peer_gm_data(rsn)[dir_idx[direction]]
        except ValueError as exc:
            msg = f'RSN {rsn} not available.'
            raise ValueError(msg) from exc
        gm_data = import_PEER(gm_filename)
        gm_dt = gm_data[1, 0] - gm_data[0, 0]
        ag = gm_data[:, 1] * scaling

    else:
        # Note: we examined CMS suites and decided not to use them.
        msg = f'Unsupported suite type: {suite_type}'
        raise NotImplementedError(msg)

    ag *= additional_scaling

    #
    # modal analysis
    #

    modal_analysis = solver.ModalAnalysis(
        mdl, {loadcase.name: loadcase}, num_modes=num_levels * 6
    )
    modal_analysis.settings.store_forces = False
    modal_analysis.settings.store_fiber = False
    modal_analysis.settings.restrict_dof = [False, True, False, True, False, True]
    modal_analysis.run()

    periods = modal_analysis.results[loadcase.name].periods
    assert periods is not None

    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # show_deformed_shape(
    #     modal_analysis, loadcase.name, 0, 0.00,
    #     extrude=False, animation=False)

    # mnstar = modal_analysis.modal_participation_factors(loadcase.name, 'x')[1]
    # np.cumsum(mnstar)

    #
    # time-history analysis
    #

    t_bar = periods[0]

    if damping == 'rayleigh':
        damping_input = {
            'type': 'rayleigh',
            'ratio': 0.02,
            'periods': [t_bar, t_bar / 10.00],
        }
    elif damping == 'modal':
        damping_input = {
            'type': 'modal+stiffness',
            'num_modes': (num_levels) * 3,
            'ratio_modal': 0.02,
            'period': t_bar / 10.00,
            'ratio_stiffness': 0.001,
        }
    else:
        msg = f'Invalid damping type: {damping}'
        raise ValueError(msg)

    identifier = '::'.join(
        [
            str(x)
            for x in [
                archetype,
                suite_type,
                hazard_level,
                gm_number,
                analysis_dt,
                direction,
                damping,
                additional_scaling,
            ]
        ]
    )
    os.makedirs('/tmp/osmg_logs/', exist_ok=True)  # noqa: PTH103
    log_file = f'/tmp/osmg_logs/{identifier}'

    nlth = solver.THAnalysis(mdl, {loadcase.name: loadcase})
    nlth.settings.log_file = log_file
    nlth.settings.restrict_dof = [False, True, False, True, False, True]
    nlth.settings.store_fiber = False
    nlth.settings.store_forces = False
    nlth.settings.store_reactions = True
    nlth.settings.store_release_force_defo = False
    nlth.settings.specific_nodes = specific_nodes

    # we want to store results at a resolution of 0.01s
    # to avoid running out of memory
    assert analysis_dt <= 0.01  # noqa: PLR2004
    skip_steps = int(0.01 / analysis_dt)

    nlth.run(
        analysis_dt,
        ag,
        None,
        None,
        gm_dt,
        damping=damping_input,
        print_progress=True,
        drift_check=0.10,  # 10% drift
        skip_steps=skip_steps,  # only save after X converged states
        time_limit=47.95,  # hours
        dampen_out_residual=True,
        finish_time=0.00,  # means run the entire file
    )

    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # show_deformed_shape(nlth, loadcase.name, 6629, 30.00, False, None, None, False)
    # show_deformed_shape(nlth, loadcase.name, 3788, 130.00, False, None, None, False)

    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # Code to plot hysteresis loops for quality assurance
    # # Added Thu Jun 20 09:59:19 AM PDT 2024
    # from osmg.graphics.preprocessing_3d import show

    # show(mdl, loadcase)
    # import osmg

    # elms = mdl.list_of_elements()
    # zerolen_elms = []
    # for elm in elms:
    #     if isinstance(elm, osmg.ops.element.ZeroLength):
    #         zerolen_elms.append(elm)
    # found = False
    # for elm in zerolen_elms:
    #     coords = elm.nodes[0].coords
    #     xcoord, ycoord, zcoord = coords
    #     # if 1199.00 <= xcoord <= 1201.00 and 0.00 <= zcoord <= 1.00:  # col base (A)
    #     # if 1187.30 <= xcoord <= 1187.40 and 167.70 <= zcoord <= 167.80:  # beam (B)
    #     # if 599.90 <= xcoord <= 600.10 and 0.0001 <= zcoord <= 1.00:  # grv col base (C)
    #     # if 599.98 <= xcoord <= 599.999 and 199.99 <= zcoord <= 180.001:  # grv bm (D)
    #     if 1187.35 <= xcoord <= 1187.45 and 179.99 <= zcoord <= 180.001:  # pz (E)
    #         found = True
    #         break
    # results = nlth.results[loadcase.name].release_force_defo[elm.uid]
    # result_vals = results.values()
    # moment = np.array([x[5] for x in result_vals])
    # # moment = np.array([x[4] for x in result_vals])
    # moment /= 1000.00 * 12.00  # to kip-ft
    # rotation = np.array([x[2] for x in result_vals])
    # # rotation = np.array([x[1] for x in result_vals])
    # time = np.array(nlth.time_vector)

    # import plotly.graph_objects as go

    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(x=rotation, y=moment, mode='lines', name='Moment vs. Rotation')
    # )
    # fig.update_layout(
    #     title='Moment vs. Rotation', xaxis_title='Rotation', yaxis_title='Moment'
    # )
    # fig.show()

    # from plotly.subplots import make_subplots

    # fig = make_subplots(
    #     rows=2, cols=1, shared_xaxes=True, subplot_titles=('Moment', 'Rotation')
    # )
    # fig.add_trace(
    #     go.Scatter(x=time, y=moment, mode='lines', name='Moment'), row=1, col=1
    # )
    # fig.add_trace(
    #     go.Scatter(x=time, y=rotation, mode='lines', name='Rotation'), row=2, col=1
    # )
    # fig.update_layout(
    #     title='Moment and Rotation vs. Index',
    #     xaxis=dict(title='Index'),
    #     yaxis=dict(title='Moment'),
    #     yaxis2=dict(title='Rotation'),
    # )
    # fig.show()
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # store response quantities

    df = pd.DataFrame()
    df['time--'] = np.array(nlth.time_vector)

    df['Rtime--'] = np.array(nlth.results[loadcase.name].clock)
    df['Rtime--'] -= df['Rtime--'].iloc[0]
    df['Subdiv--'] = np.array(nlth.results[loadcase.name].subdivision_level)

    if direction == 'x':
        j = 1
    elif direction == 'y':
        j = 2
    else:
        msg = f'Invalid direction: {direction}'
        raise ValueError(msg)

    for lvl in range(num_levels + 1):
        df[f'FA-{lvl}-{j}'] = nlth.retrieve_node_abs_acceleration(
            lvl_nodes[lvl], loadcase.name
        ).loc[:, 'abs ax']
        df[f'FV-{lvl}-{j}'] = nlth.retrieve_node_abs_velocity(
            lvl_nodes[lvl], loadcase.name
        ).loc[:, 'abs vx']
        if lvl > 0:
            us = nlth.retrieve_node_displacement(lvl_nodes[lvl], loadcase.name).loc[
                :, 'ux'
            ]
            if lvl == 1:
                dr = us / level_heights[lvl - 1]
            else:
                us_prev = nlth.retrieve_node_displacement(
                    lvl_nodes[lvl - 1], loadcase.name
                ).loc[:, 'ux']
                dr = (us - us_prev) / level_heights[lvl - 1]
            df[f'ID-{lvl}-{j}'] = dr

    df[f'Vb-0-{j}'] = nlth.retrieve_base_shear(loadcase.name)[:, 0]

    df.columns = pd.MultiIndex.from_tuples(
        [x.split('-') for x in df.columns.to_list()]
    )
    df = df.sort_index(axis=1)

    df = df.set_index('time')
    df = df[~df.index.duplicated()]

    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # import plotly.graph_objects as go
    # fig = go.Figure()
    # for column in df['ID'].columns:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=df['ID'].index,
    #             y=df['ID'][column],
    #             mode='lines',
    #             name='-'.join(column),
    #         )
    #     )
    # fig.update_layout(
    #     title='Time Series Data', xaxis_title='Time', yaxis_title='Values'
    # )
    # fig.show()
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # interpoate to a time step of 0.01 (to save space)
    # # and set index to `time`
    # step = 0.01
    # new_index = np.arange(df.index.min(), df.index.max() + step, step)
    # df_resampled = pd.DataFrame(index=new_index, columns=df.columns)
    # df_resampled.index.name = df.index.name
    # for col in df.columns:
    #     df_resampled[col] = np.interp(new_index, df.index.values, df[col].values)

    # # add the results to the database
    # if not os.path.isdir(
    #     f'extra/structural_analysis/results/{sub_path}'
    # ):  # noqa: PTH112, RUF100
    #     os.makedirs(f'extra/structural_analysis/results/{sub_path}')
    # db_handler = DB_Handler(
    #     db_path=f'extra/structural_analysis/results/{sub_path}results_{group_id}.sqlite'
    # )
    # try:
    #     db_handler.store_data(
    #         identifier=identifier,
    #         dataframe=df_resampled,
    #         metadata=info,
    #         log_content=log_contents,
    #     )
    # except:
    #     # if it fails *for any reason*, pickle the result variables and save them
    #     # with a unique name
    #     out = {
    #         'identifier': identifier,
    #         'dataframe': df_resampled,
    #         'metadata': info,
    #         'log_content': log_contents,
    #     }
    #     with open(
    #         Path(f'extra/structural_analysis/results/{sub_path}{identifier}'), 'wb'
    #     ) as f:
    #         pickle.dump(out, f)

    # # add EDP results to the database
    # edp_db_handler = DB_Handler(
    #     db_path=(
    #         f'extra/structural_analysis/results/'
    #         f'{sub_path}edp_results_{group_id}.sqlite'
    #     )
    # )
    # edps = obtain_edps(df)
    # try:
    #     edp_db_handler.store_data(identifier, edps, '', '')
    # except:
    #     # if it fails *for any reason*, pickle the result variables and save them
    #     # with a unique name
    #     out = {
    #         'identifier': identifier,
    #         'dataframe': df_resampled,
    #         'metadata': info,
    #         'log_content': log_contents,
    #     }
    #     with open(
    #         Path(f'extra/structural_analysis/results/{sub_path}edp_{identifier}'),
    #         'wb',
    #     ) as f:
    #         pickle.dump(out, f)


if __name__ == '__main__':
    main()
