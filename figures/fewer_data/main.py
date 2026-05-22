"""
# Model performance with limited data.

## Objective

See how the model performs when fitted to smaller samples.

"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import models
from src.handle_data import load_dataset, remove_collapse, retrieve_values


mpl.rc_file('figures/matplotlibrc')


OUTPUT_PATH = Path(
    'figures/fewer_data/figure.pdf'
)

SYSTEM = 'smrf'
STORIES = '9'
RC = 'ii'
STORY = 'max-max'

SEED = 32
CENSORING_LIMIT = 0.00025

X_LIMITS = (0, 0.04)
Y_LIMITS = (0, 0.025)

X_MAJOR_TICKS = np.arange(0.00, 0.041, 0.01)
X_MINOR_TICKS = np.arange(0.00, 0.041, 0.005)
Y_MAJOR_TICKS = np.arange(0.00, 0.026, 0.01)
Y_MINOR_TICKS = np.arange(0.00, 0.026, 0.005)

SUBSETS = (
    {
        'title': '13 Analyses\n2 Hazard Levels',
        'num_per_hz': 13,
        'hazard_levels': ['9', '17'],
    },
    {
        'title': '20 Analyses\n6 Hazard Levels',
        'num_per_hz': 20,
        'hazard_levels': ['1', '5', '9', '13', '17', '21'],
    },
    {
        'title': 'Full Sample\n40 Analyses\n25 Hazard Levels',
        'num_per_hz': None,
        'hazard_levels': None,
    },
)


def configure_rcparams() -> None:
    plt.rcParams.update(
        {
            'font.size': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'font.family': 'serif',
            'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
        }
    )


def percent_labels(values: np.ndarray) -> list[str]:
    return [f'{int(100 * value)}%' for value in values]


def fit_model(data: pd.DataFrame) -> models.Model_Trilinear_Weibull:
    model = models.Model_Trilinear_Weibull()
    model.add_data(data['PID'].to_numpy(), data['RID'].to_numpy())
    model.censoring_limit = CENSORING_LIMIT
    model.fit(method='mle-fast', global_search=True)
    print(model.parameter_names)
    print(model.parameter_bounds)
    print(model.parameters)
    print()
    return model


def subset_data(
    data: pd.DataFrame,
    hazard_levels: list[str] | None,
    num_per_hz: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if hazard_levels is None or num_per_hz is None:
        return data.copy()

    subset_frames = []
    for hz in hazard_levels:
        hz_data = data.loc[data.index.get_level_values('hz') == hz]
        n_available = len(hz_data)
        n_select = min(num_per_hz, n_available)
        sampled_rows = rng.choice(hz_data.index.to_numpy(), size=n_select, replace=False)
        subset_frames.append(hz_data.loc[sampled_rows])

    return pd.concat(subset_frames).sort_index()


def plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    *,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    model = fit_model(data)


    n = len(data)

    if n < 60:
        model.calculate_rolling_quantiles(fraction=0.45)
    elif n < 200:
        model.calculate_rolling_quantiles(fraction=0.2)
    else:
        model.calculate_rolling_quantiles(fraction=0.075)    

    scatter = ax.scatter(
        data['PID'].to_numpy(),
        data['RID'].to_numpy(),
        s=2.2,
        facecolors='none',
        edgecolors='black',
        linewidths=0.30,
        alpha=0.15,
        label='Simulation data' if show_legend else None,
    )

    roll50, = ax.plot(
        model.rolling_pid,
        model.rolling_rid_50,
        color='tab:red',
        linewidth=1.0,
        label='Rolling quantiles' if show_legend else None,
    )
    ax.plot(
        model.rolling_pid,
        model.rolling_rid_20,
        color='tab:red',
        linewidth=0.50,
    )
    ax.plot(
        model.rolling_pid,
        model.rolling_rid_80,
        color='tab:red',
        linewidth=0.50,
    )

    model_pid = np.linspace(0.00, 0.08, 1000)
    model_rid_50 = model.evaluate_inverse_cdf(0.50, model_pid)
    model_rid_20 = model.evaluate_inverse_cdf(0.20, model_pid)
    model_rid_80 = model.evaluate_inverse_cdf(0.80, model_pid)

    fit50, = ax.plot(
        model_pid,
        model_rid_50,
        color='C0',
        linestyle='dashed',
        linewidth=1.0,
        label='Fitted model' if show_legend else None,
    )
    ax.plot(
        model_pid,
        model_rid_20,
        color='C0',
        linestyle='dashed',
        linewidth=0.50,
    )
    ax.plot(
        model_pid,
        model_rid_80,
        color='C0',
        linestyle='dashed',
        linewidth=0.50,
    )

    # ax.set_title(title, fontsize=5.0)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)

    ax.set_xticks(X_MAJOR_TICKS)
    ax.set_xticks(X_MINOR_TICKS, minor=True)
    ax.set_yticks(Y_MAJOR_TICKS)
    ax.set_yticks(Y_MINOR_TICKS, minor=True)

    ax.set_xticklabels(percent_labels(X_MAJOR_TICKS))
    ax.set_yticklabels(percent_labels(Y_MAJOR_TICKS))

    ax.grid(which='major', alpha=0.5, linewidth=0.6)
    ax.grid(which='minor', alpha=0.2, linewidth=0.4)

    ax.set_xlabel('PSDR')
    ax.set_ylabel('RSDR' if show_ylabel else '')

    ax.text(
        0.05,
        0.95,
        f'n={len(data)}\n{title}',
        ha='left',
        va='top',
        transform=ax.transAxes,
        fontsize=6.0,
    )

    if show_legend:
        return scatter, roll50, fit50

    return None


def main() -> None:
    configure_rcparams()
    rng = np.random.default_rng(SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_data = remove_collapse(
        load_dataset('extra/structural_analysis/results/data.parquet')
    )

    data = retrieve_values(SYSTEM, STORIES, RC, STORY, all_data, apply_abs=True)
    data = data[['PID', 'RID']]
    data = data.loc[data.index.get_level_values('dr') == 'x']

    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(4.85, 1.6), sharex=True, sharey=True)

    legend_handles = None
    for i, (ax, subset_spec) in enumerate(zip(axs, SUBSETS, strict=False)):
        panel_data = subset_data(
            data,
            subset_spec['hazard_levels'],
            subset_spec['num_per_hz'],
            rng,
        )

        handles = plot_panel(
            ax,
            panel_data,
            subset_spec['title'],
            show_ylabel=(i == 0),
            show_legend=(i == 2),
        )

        if handles is not None:
            legend_handles = handles

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ['Simulation data', 'Rolling quantiles', 'Fitted model'],
            loc='upper center',
            ncols=3,
            frameon=False,
            fontsize=6,
        )

    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.10, right=0.98, wspace=0.18)
    fig.savefig(OUTPUT_PATH)


if __name__ == '__main__':
    main()
