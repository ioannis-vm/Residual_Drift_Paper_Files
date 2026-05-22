"""
Story-by-story v.s. max-of-all-stories

Investigating the equivalence of the story-by-story
v.s. max-of-all-stories approach in determining the likelihood of
irreparable damage.

Written: Wed Sep 11 06:08:01 AM PDT 2024
by Ioannis Vouvakis Manousakis of UC Berkeley.
Modified: Sun Oct  6 04:05:44 PM PDT 2024

"""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm

from src import models
from src.handle_data import load_dataset, remove_collapse


mpl.rc_file('figures/matplotlibrc')


SYSTEMS = ('smrf', 'scbf', 'brbf')
STORIES = ('9',)
RCS = ('ii', 'iv')

FIGSIZE = (4.85, 2.0)
N_ROWS = 2
N_COLUMNS = 3

X_MAJOR_TICKS = np.arange(0.00, 0.051, 0.01)
X_MINOR_TICKS = np.arange(0.00, 0.051, 0.005)

X_LIMITS = (0.0, 0.04)
Y_LIMITS = (0.0, 1.0)

OUTPUT_DIR = Path(
    'figures/max_max_equivalent/'
)
OUTPUT_FILE = OUTPUT_DIR / 'figure.pdf'


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


def iter_cases():
    yield from product(SYSTEMS, STORIES, RCS)


def estimate_corr(x_i: float, x_j: float, params: tuple[float, float]) -> float:
    c_1, c_2 = params
    return np.exp(-c_1 * np.abs(x_i - x_j) ** c_2)


def style_axis(ax) -> None:
    ax.xaxis.set_major_locator(mticker.FixedLocator(X_MAJOR_TICKS))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(X_MINOR_TICKS))

    ax.set_xticklabels([f'{int(t * 100)}%' for t in X_MAJOR_TICKS])

    ax.grid(which='major', alpha=0.5, linewidth=0.6)
    ax.grid(which='minor', alpha=0.2, linewidth=0.4)

    ax.set(xlim=X_LIMITS, ylim=Y_LIMITS)


def main() -> None:
    configure_rcparams()
    np.random.seed(32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data = remove_collapse(
        load_dataset(Path('extra/structural_analysis/results/data.parquet'))
    )

    plt.close()
    fig, axs = plt.subplots(
        N_ROWS,
        N_COLUMNS,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )

    for idx, (system, stories, rc) in enumerate(tqdm(list(iter_cases()))):
        ax = axs.flat[idx]

        data = (
            all_data.loc[(system, stories, rc), ['PID', 'RID']]
            .dropna(how='all')
            .abs()
        )

        num_stories = int(stories)

        hazard_levels = ['16', '17', '18'] if rc == 'ii' else ['23', '24', '25']

        data_hz = data.loc[(hazard_levels,), :]

        empirical_max_rid = data_hz['RID'].unstack('loc').max(axis=1)

        data_maxmax = data_hz.groupby(['hz', 'gm', 'dr']).max()

        model_maxmax = models.Model_Trilinear_Weibull()
        model_maxmax.add_data(
            data_maxmax['PID'].to_numpy(), data_maxmax['RID'].to_numpy()
        )
        model_maxmax.censoring_limit = 0.005
        model_maxmax.fit(method='mle-fast', global_search=True)

        story_models = {}
        for story in data.index.get_level_values('loc').unique():
            model_story = models.Model_Trilinear_Weibull()
            data_story = data_hz[data_hz.index.get_level_values('loc') == story]

            model_story.add_data(
                data_story['PID'].to_numpy(), data_story['RID'].to_numpy()
            )
            model_story.censoring_limit = 0.005
            model_story.fit(method='mle-fast', global_search=True)

            story_models[story] = model_story

        num_realizations = 1000

        pids = data.loc[(hazard_levels,), 'PID'].unstack('loc')
        bootstrap_idx = np.random.choice(len(pids), size=num_realizations)

        pid_sample = pd.DataFrame(
            pids.to_numpy()[bootstrap_idx, :], columns=pids.columns
        )

        rids = data.loc[(hazard_levels,), 'RID'].unstack('loc')

        copula_correlations = np.eye(num_stories)

        story_heights = [15.0]
        for _ in range(num_stories - 1):
            story_heights.append(story_heights[-1] + 13.0)

        story_heights = np.array(story_heights)
        story_heights /= story_heights[-1]

        for i_story, j_story in combinations(
            data.index.get_level_values('loc').unique(), 2
        ):
            x_i = story_heights[int(i_story) - 1]
            x_j = story_heights[int(j_story) - 1]

            val = estimate_corr(x_i, x_j, (10.0, 2.0))
            copula_correlations[int(i_story) - 1, int(j_story) - 1] = val
            copula_correlations[int(j_story) - 1, int(i_story) - 1] = val

        l_mat = np.linalg.cholesky(copula_correlations)

        u = np.random.uniform(0.0, 1.0, size=(num_realizations, num_stories))
        z = norm.ppf(u)
        z_corr = (l_mat @ z.T).T
        u_corr = norm.cdf(z_corr)

        story_rids = {}
        for story in data.index.get_level_values('loc').unique():
            model_story = story_models[story]
            model_story.uniform_sample = u_corr[:, int(story) - 1]

            story_rids[story] = model_story.generate_rid_samples(
                pid_sample[story].to_numpy()
            )

        max_rid_from_stories = pd.DataFrame(story_rids).max(axis=1)

        rid_maxmax = model_maxmax.generate_rid_samples(pid_sample.max(axis=1))

        label_sa = 'Structural Analysis Results' if idx == 1 else None
        label_story = 'Story-By-Story' if idx == 1 else None
        label_max = 'Spatial Maxima' if idx == 1 else None

        sns.ecdfplot(empirical_max_rid, ax=ax, label=label_sa, color='black')
        sns.ecdfplot(
            max_rid_from_stories,
            ax=ax,
            label=label_story,
            color='C0',
            linestyle='dashed',
        )
        sns.ecdfplot(
            rid_maxmax,
            ax=ax,
            label=label_max,
            color='C1',
            linestyle='dotted',
        )

        style_axis(ax)

        ax.set_ylabel('CDF')
        ax.text(
            0.95,
            0.05,
            f'{system.upper()}-{stories.upper()}-{rc.upper()}',
            ha='right',
            va='bottom',
            transform=ax.transAxes,
        )

    axs[1, 1].set_xlabel('RSDR')

    fig.legend(ncols=3, loc='upper center', frameon=False)
    fig.subplots_adjust(
        wspace=0.20,
        hspace=0.20,
        bottom=0.18,
        right=0.98,
        left=0.095,
        top=0.85,
    )

    plt.show()
    # plt.savefig(OUTPUT_FILE)


if __name__ == '__main__':
    main()
