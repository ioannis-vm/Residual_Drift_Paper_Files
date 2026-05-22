"""
Pushover Analysis.

Perform a series of pushover analyses trying to understand how
residual drift is affected by various modeling assumptions, such as:
- The presence / absence of the gravity framing
- The way the lateral load-resisting beams are connected to the columns.
We push the building with a modal distribution of loads until it
reaches a drift of 2% and then slowly reduce the imposed loads to
monitor the residual deformation.

Written: Thu Jul 25 03:25:18 PM PDT 2024
By Ioannis Vouvakis Manousakis of UC Berkeley.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osmg import solver
from osmg.common import G_CONST_IMPERIAL
from osmg.gen.query import ElmQuery

from extra.structural_analysis.src.structural_analysis.archetypes_2d import brbf_1_ii


def main() -> None:  # noqa: C901, D103, RUF100
    target_drifts = (1.00, 2.00, 3.00, 4.00, 5.00)

    for target_drift in target_drifts:
        direction = 'x'
        mdl, loadcase = brbf_1_ii(direction, pinned_beams=False)

        # # run a static analysis and plot the basic forces
        # static_anl = solver.StaticAnalysis(mdl, {loadcase.name: loadcase})
        # static_anl.run()
        # from osmg.graphics.postprocessing_3d import show_basic_forces
        # camera = {
        #     'up': {'x': 0, 'y': 0, 'z': 1},
        #     'center': {'x': 0, 'y': 0, 'z': 0},
        #     'eye': {'x': 0.00, 'y': -1.00, 'z': 0.00},
        #     'projection': {'type': 'orthographic'},
        # }
        # metadata = show_basic_forces(
        #     static_anl,
        #     loadcase.name,
        #     0,
        #     1.00,
        #     0.00,
        #     0.00,
        #     0.00,
        #     0.00,
        #     10,
        #     1e-3,
        #     1.00/1e3/12.00,
        #     camera=camera,
        # )

        # # # Visualize the model
        # from osmg.graphics.preprocessing_3d import show
        # camera = {
        #     'up': {'x': 0, 'y': 0, 'z': 1},
        #     'center': {'x': 0, 'y': 0, 'z': 0},
        #     'eye': {'x': 0.00, 'y': -1.00, 'z': 0.00},
        #     'projection': {'type': 'orthographic'},
        # }
        # show(mdl, loadcase, camera=camera)

        num_levels = len(mdl.levels) - 1
        level_heights = np.diff([level.elevation for level in mdl.levels.values()])

        lvl_nodes = []
        base_node = next(iter(mdl.levels[0].nodes.values()))
        lvl_nodes.append(base_node)
        for i in range(num_levels):
            lvl_nodes.append(loadcase.parent_nodes[i + 1])  # noqa: PERF401

        # modal analysis (to get the mode shape)

        # fix leaning column
        elmq = ElmQuery(mdl)
        for i in range(num_levels):
            lc_node = elmq.search_node_lvl(0.00, 0.00, i + 1)
            assert lc_node is not None
            lc_node.restraint = [False, False, False, True, True, True]

        modal_analysis = solver.ModalAnalysis(
            mdl, {loadcase.name: loadcase}, num_modes=10
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
        modal_analysis.results[loadcase.name].periods  # noqa: B018

        modeshape_lst = []
        for lc_node in lvl_nodes:
            modeshape_lst.append(  # noqa: PERF401
                modal_analysis.results[loadcase.name].node_displacements[
                    lc_node.uid
                ][0][0]
            )
        modeshape = np.array(modeshape_lst)

        # show_deformed_shape(modal_analysis, loadcase.name, 0, 0.00, extrude=True, camera=camera)

        # pushover analysis
        for i in range(num_levels):
            lc_node = elmq.search_node_lvl(0.00, 0.00, i + 1)
            assert lc_node is not None
            lc_node.restraint = [False, False, False, False, False, False]

        # define analysis
        anl = solver.PushoverAnalysis(mdl, {loadcase.name: loadcase})
        anl.settings.store_forces = True
        anl.settings.store_release_force_defo = False
        anl.settings.solver = 'SparseSYM'
        anl.settings.restrict_dof = [False, True, False, True, False, True]
        control_node = lvl_nodes[-1]

        anl.run(
            'x',
            [target_drift / 100.00 * 180.00, None],
            control_node,
            0.10,
            modeshape=modeshape,
        )

        res_df = pd.DataFrame()
        for i_story, node in enumerate(lvl_nodes):
            if i_story == 0:
                continue
            results = np.column_stack(
                anl.table_pushover_curve(loadcase.name, 'x', node)
            )
            if i_story == 1:
                res_df['Vb'] = results[:, 1]
            res_df[f'Level {i_story}'] = results[:, 0]
        res_df.index.name = 'Step'

        res_df['Drift 1'] = (res_df['Level 1'] / level_heights[0]) * 100.00
        for i in range(2, num_levels + 1):
            res_df[f'Drift {i}'] = (
                (res_df[f'Level {i}'] - res_df[f'Level {i - 1}'])
                / level_heights[i - 1]
            ) * 100.00

        res_df['Global'] = (
            np.column_stack(
                anl.table_pushover_curve(loadcase.name, 'x', lvl_nodes[-1])
            )[:, 0]
            / np.sum(level_heights)
            * 100.00
        )

        res_df.to_parquet(f'/tmp/fixed_{target_drift}')

    plt.close()
    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    for connectivity in ('pinned', 'fixed'):
        for target_drift in target_drifts:
            data = pd.read_parquet(f'/tmp/{connectivity}_{target_drift}')

            ax.plot(
                data['Global'],
                data['Vb'] / 1e3,
                label=f'{connectivity} beams',
            )
            print(
                connectivity,
                target_drift,
                f'{data["Drift 1"].max():.3f}',
                f'{data["Drift 1"].iloc[-1]:.3f}',
            )

    ax.set(xlabel='Story drift (%)', ylabel='Base shear (kips)')
    ax.grid(which='both', linewidth=0.30)
    # ax.set(xlim=(-0.20, 7.70), ylim=(-2.00, 49.00))
    fig.tight_layout()
    plt.savefig('/tmp/fig.png', dpi=1200)
    plt.show()

    # # metadata = show_deformed_shape(
    # #     anl,
    # #     loadcase.name,
    # #     anl.results[loadcase.name].n_steps_success - 1,
    # #     # 40,
    # #     5.0,
    # #     extrude=True,
    # #     animation=False,
    # #     camera=camera,
    # # )

    # from osmg.graphics.postprocessing_3d import show_deformed_shape

    # metadata = show_deformed_shape(
    #     anl,
    #     loadcase.name,
    #     anl.results[loadcase.name].n_steps_success - 1,
    #     # 40,
    #     5.0,
    #     extrude=False,
    #     animation=True,
    #     camera=camera,
    #     step_skip=1,
    # )

    # from osmg.graphics.postprocessing_3d import show_basic_forces
    # metadata = show_basic_forces(
    #     anl,
    #     loadcase.name,
    #     0,
    #     1.00,
    #     0.00,
    #     0.00,
    #     0.00,
    #     0.00,
    #     10,
    #     1e-3,
    #     1.00/1e3/12.00,
    #     camera=camera,
    # )


if __name__ == '__main__':
    main()
