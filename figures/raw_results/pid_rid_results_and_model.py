"""
Code to generate figures for the journal paper.
Title: Results for a single archetype, before fitting the model.
Written: Fri Oct  4 09:42:12 AM PDT 2024
By Ioannis Vouvakis Manousakis of UC Berkeley.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from src import models
from src.handle_data import load_dataset, remove_collapse


# Load project-specific matplotlib settings
mpl.rc_file('figures/matplotlibrc')


def format_percent_ticks(values: list[float], labeled: set[float]) -> list[str]:
    """
    Return tick labels in percent format, leaving some unlabeled.

    Parameters
    ----------
    values : list of float
        Tick locations expressed as ratios, e.g. 0.01 for 1%.
    labeled : set of float
        Subset of `values` to label.

    Returns
    -------
    list of str
        Tick labels.
    """
    labels = []
    for value in values:
        if value in labeled:
            labels.append(f'{value * 100:.0f}%')
        else:
            labels.append('')
    return labels


def main():
    """
    Generate a scatter plot of residual versus peak story drift ratio.

    Returns
    -------
    None
    """

    plt.rcParams.update(
        {
            # 'text.usetex': True,
            'font.size': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'font.family': 'serif',
            'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
        }
    )

    output_path = Path('figures/raw_results')
    output_path.mkdir(parents=True, exist_ok=True)

    # Retrieve all data
    all_data = remove_collapse(
        load_dataset(Path('extra/structural_analysis/results/data.parquet'))
    )

    system, stories, rc = 'scbf', '3', 'ii'
    story = '1'

    data = (
        all_data.loc[(system, stories, rc), ['PID', 'RID']].dropna(how='all').abs()
    )
    data_sub = data[data.index.get_level_values('loc') == story]

    # Axis limits
    x_max = 4.00 / 100.00
    y_max = 2.00 / 100.00

    # Tick locations
    x_ticks = [0.00, 0.01, 0.02, 0.03, 0.04]
    y_ticks = [0.00, 0.01, 0.02]

    # Tick labels: x labels only at 0, 2, 4; y labels at 0, 1, 2
    x_ticklabels = format_percent_ticks(x_ticks, labeled={0.00, 0.02, 0.04})
    y_ticklabels = format_percent_ticks(y_ticks, labeled={0.00, 0.01, 0.02})

    hz_vals = sorted(data_sub.index.get_level_values('hz').unique())
    dr_vals = sorted(data_sub.index.get_level_values('dr').unique())

    highlight_hz_colors = {
        '9': 'tab:blue',
        '17': 'tab:orange',
    }
    dr_markers = {'x': 'o', 'y': '^'}

    plt.close('all')
    fig, axs = plt.subplots(1, 3, figsize=(4.85, 1.6))

    for hz in hz_vals:
        is_highlight = hz in highlight_hz_colors
        color = highlight_hz_colors.get(hz, 'black')

        for dr in dr_vals:
            subset = data_sub.xs((hz, dr), level=('hz', 'dr'))

            axs[0].scatter(
                subset['PID'],
                subset['RID'],
                s=2.2,
                marker=dr_markers[dr],
                facecolors=color if is_highlight else 'none',
                edgecolors=color,
                linewidths=0.30,
                alpha=0.25 if is_highlight else 0.15,
            )

    axs[0].set_xlim(0.0, x_max)
    axs[0].set_ylim(0.0, y_max)

    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_ticklabels)

    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_ticklabels)

    axs[0].set_xlabel('PSDR')
    axs[0].set_ylabel('RSDR')

    axs[1].scatter(
        data_sub['PID'],
        data_sub['RID'],
        s=2.2,
        marker='o',
        facecolors='none',
        edgecolors='black',
        linewidths=0.30,
        alpha=0.15,
    )

    axs[1].set_xlim(0.0, x_max)
    axs[1].set_ylim(0.0, y_max)

    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_ticklabels)

    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_ticklabels)

    axs[1].set_xlabel('PSDR')

    roll = models.Model()
    roll.add_data(data_sub['PID'].to_numpy(), data_sub['RID'].to_numpy())
    roll.calculate_rolling_quantiles()
    roll_pid = roll.rolling_pid
    roll_rid_50 = roll.rolling_rid_50
    roll_rid_20 = roll.rolling_rid_20
    roll_rid_80 = roll.rolling_rid_80
    axs[1].plot(
        roll_pid,
        roll_rid_50,
        color='black',
    )
    axs[1].plot(
        roll_pid,
        roll_rid_20,
        linewidth=0.50,
        color='black',
    )
    axs[1].plot(
        roll_pid,
        roll_rid_80,
        linewidth=0.50,
        color='black',
    )

    fema = models.Model_P58()
    fema_rid = fema.delta_fnc(pid=roll_pid, delta_y=0.00424)
    axs[1].plot(
        roll_pid,
        fema_rid,
        color='C3',
        linestyle='dashed',
    )

    axs[2].set_xlim(0.0, x_max)
    axs[2].set_ylim(0.0, y_max)

    axs[2].set_xticks(x_ticks)
    axs[2].set_xticklabels(x_ticklabels)

    axs[2].set_yticks(y_ticks)
    axs[2].set_yticklabels(y_ticklabels)

    axs[2].set_xlabel('PSDR')

    model = models.Model_Trilinear_Weibull()
    model.add_data(data_sub['PID'].to_numpy(), data_sub['RID'].to_numpy())
    model.censoring_limit = 0.00025
    model.fit(method='mle-fast', global_search=False)
    model_pid = np.linspace(0.00, 0.04, 1000)
    model_rid_50 = model.evaluate_inverse_cdf(0.50, model_pid)
    model_rid_20 = model.evaluate_inverse_cdf(0.20, model_pid)
    model_rid_80 = model.evaluate_inverse_cdf(0.80, model_pid)
    axs[2].plot(
        model_pid,
        model_rid_50,
        color='C0',
    )
    axs[2].plot(
        model_pid,
        model_rid_20,
        linewidth=0.50,
        color='C0',
    )
    axs[2].plot(
        model_pid,
        model_rid_80,
        linewidth=0.50,
        color='C0',
    )


    plt.tight_layout(pad=0.10)

    fig.savefig(output_path / 'pid_rid_results_and_model_matplotlib.pdf')
    fig.savefig(output_path / 'pid_rid_results_and_model_matplotlib.svg')
    # plt.show()


if __name__ == '__main__':
    main()
