# """
# Title: Model parameters.
# Written: Sun Oct  6 07:17:58 AM PDT 2024
# By Ioannis Vouvakis Manousakis of UC Berkeley.
# """

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import models
from src.handle_data import load_dataset, remove_collapse


mpl.rc_file('figures/matplotlibrc')


SYSTEMS = ('smrf', 'scbf', 'brbf')
RCS = ('ii', 'iv')
STORY_COUNTS = ('3', '6', '9')

X_MAJOR_TICKS = np.arange(0.00, 0.071, 0.02)
Y_MAJOR_TICKS = np.arange(0.00, 0.071, 0.01)
MINOR_TICKS = np.arange(0.00, 0.071, 0.005)

FIGSIZE = (7.25, 4.85)
N_ROWS = 3
N_COLUMNS = 6

X_LIMITS = (0.0, 0.06)
Y_LIMITS = (0.0, 0.025)

OUTPUT_PATH = Path(
    'figures/model_parameters/'
    'fit_all_archetypes.pdf'
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


def iter_archetypes():
    yield from product(SYSTEMS, RCS, STORY_COUNTS)


def plot_data(rid_model, ax=None, scatter_kwargs=None) -> None:  # noqa: ANN001
    if scatter_kwargs is None:
        scatter_kwargs = {
            's': 2.2,
            'facecolor': 'none',
            'edgecolor': 'black',
            'alpha': 0.05,
        }

    if ax is None:
        _, ax = plt.subplots()
        ax.scatter(rid_model.raw_pid, rid_model.raw_rid, **scatter_kwargs)
        plt.show()
        return

    ax.scatter(rid_model.raw_pid, rid_model.raw_rid, **scatter_kwargs)


def plot_model(
    rid_model,
    ax,
    rolling: bool = True,
    training: bool = True,
    model: bool = True,
    model_color: str = 'C0',
) -> None:
    if rid_model.fit_status == 'False':
        rid_model.fit()

    if training:
        plot_data(rid_model, ax=ax)

    if rolling:
        rid_model.calculate_rolling_quantiles()

        ax.plot(rid_model.rolling_pid, rid_model.rolling_rid_50, 'tab:red')
        ax.plot(
            rid_model.rolling_pid,
            rid_model.rolling_rid_20,
            'tab:red',
            linewidth=0.50,
        )
        ax.plot(
            rid_model.rolling_pid,
            rid_model.rolling_rid_80,
            'tab:red',
            linewidth=0.50,
        )

    if model:
        model_pid = np.linspace(0.00, 0.08, 1000)
        model_rid_50 = rid_model.evaluate_inverse_cdf(0.50, model_pid)
        model_rid_20 = rid_model.evaluate_inverse_cdf(0.20, model_pid)
        model_rid_80 = rid_model.evaluate_inverse_cdf(0.80, model_pid)

        ax.plot(model_pid, model_rid_50, model_color)
        ax.plot(model_pid, model_rid_20, model_color, linestyle='dashed')
        ax.plot(model_pid, model_rid_80, model_color, linestyle='dashed')


def build_models(all_data) -> tuple[dict, dict]:
    results = {}
    all_models = {}

    for system, rc, stories in tqdm(list(iter_archetypes())):
        data = (
            all_data.loc[(system, stories, rc), ['PID', 'RID']]
            .dropna(how='all')
            .abs()
        )
        data_subset = data.groupby(['hz', 'gm', 'dr']).max()

        model = models.Model_Trilinear_Weibull()
        model.add_data(data_subset['PID'].to_numpy(), data_subset['RID'].to_numpy())
        model.censoring_limit = 0.0005
        model.fit(method='mle-fast', global_search=True)

        results[f'{system}-{stories}-{rc}'.upper()] = [
            f'{model.parameters[0] * 100.00:.2f}%',
            f'{model.parameters[1]:.3f}',
            f'{(model.parameters[2] + model.parameters[0]) * 100.00:.2f}%',
            f'{model.parameters[3]:.3f}',
            f'{model.parameters[4]:.2f}',
            f'{model.parameters[5]:.2f}',
            f'{-model.fit_meta.fun:.2e}',
        ]
        all_models[(system, stories, rc)] = model

    return results, all_models


def format_parameter_text(model, system: str, stories: str, rc: str) -> str:
    x0 = f'{model.parameters[0] * 100.00:.2f}%'
    m1 = f'{model.parameters[1]:.3f}'
    x1 = f'{(model.parameters[0] + model.parameters[2]) * 100.00:.2f}%'
    m2 = f'{model.parameters[3]:.3f}'
    k1 = f'{model.parameters[4]:.2f}'
    k2 = f'{model.parameters[5]:.2f}'

    return (
        f'{system.upper()}-{stories.upper()}-{rc.upper()}\n'
        f'logL={-model.fit_meta.fun:.3e},\n'
        f'$x_0$={x0},\n'
        f'$x_1$={x1},\n'
        f'$m_1$={m1},\n'
        f'$m_2$={m2},\n'
        f'$k_1$={k1},\n'
        f'$k_2$={k2}\n'
    )


def style_axis(ax) -> None:
    ax.xaxis.set_major_locator(mticker.FixedLocator(X_MAJOR_TICKS))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(MINOR_TICKS))
    ax.yaxis.set_major_locator(mticker.FixedLocator(Y_MAJOR_TICKS))
    ax.yaxis.set_minor_locator(mticker.FixedLocator(MINOR_TICKS))

    ax.set_xticklabels([f'{int(t * 100)}%' for t in X_MAJOR_TICKS])
    ax.set_yticklabels([f'{int(t * 100)}%' for t in Y_MAJOR_TICKS])

    ax.grid(which='major', alpha=0.5, linewidth=0.6)
    ax.grid(which='minor', alpha=0.2, linewidth=0.4)
    ax.set(xlim=X_LIMITS, ylim=Y_LIMITS)


def main() -> None:
    configure_rcparams()
    np.random.seed(32)

    all_data = remove_collapse(
        load_dataset(Path('extra/structural_analysis/results/data.parquet'))
    )

    results, all_models = build_models(all_data)
    res_df = pd.DataFrame(results.values(), index=results.keys())
    # print(res_df.to_latex())

    plt.close()
    fig, axs = plt.subplots(
        N_ROWS,
        N_COLUMNS,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )

    for ax, (system, rc, stories) in zip(axs.flat, iter_archetypes(), strict=False):
        model = all_models[(system, stories, rc)]

        plot_model(model, ax=ax)
        ax.text(
            0.05,
            0.96,
            format_parameter_text(model, system, stories, rc),
            va='top',
            ha='left',
            fontsize=5.0,
            transform=ax.transAxes,
        )
        style_axis(ax)

    plt.subplots_adjust(
        left=0.09,
        top=0.98,
        bottom=0.10,
        right=0.98,
        wspace=0.20,
        hspace=0.20,
    )
    fig.text(0.5, 0.02, 'max PSDR', ha='center')
    fig.text(0.02, 0.5, 'max RSDR', va='center', rotation='vertical')
    plt.savefig(OUTPUT_PATH)


if __name__ == '__main__':
    main()
