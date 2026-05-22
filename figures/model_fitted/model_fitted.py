"""
Code to generate figures for the journal paper.
Title: Model fitted to the structural analysis results
Written: Mon Sep 30 04:32:03 PM PDT 2024
By Ioannis Vouvakis Manousakis of UC Berkeley.
"""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import fsolve
from scipy.special import gamma
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm

from src import models
from src.handle_data import load_dataset, remove_collapse

# Load project-specific matplotlib settings
mpl.rc_file('figures/matplotlibrc')


def main():
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

    seed_value = 32
    np.random.seed(seed_value)
    figpath = Path(
        'figures/model_fitted/figures'
    )
    if not figpath.exists():
        figpath.mkdir(parents=True)

    # Retrieve all data
    all_data = remove_collapse(
        load_dataset(Path('extra/structural_analysis/results/data.parquet'))
    )

    system, stories, rc = 'smrf', '3', 'ii'
    # system, stories, rc = 'brbf', '6', 'ii'

    story = '1'

    data = (
        all_data.loc[(system, stories, rc), ['PID', 'RID']].dropna(how='all').abs()
    )
    num_stories = int(stories)
    # data_subset = data.groupby(['hz', 'gm', 'dr']).max()
    data_subset = data.loc[data.index.get_level_values('loc') == story]
    model_wb = models.Model_Trilinear_Weibull()
    model_wb.add_data(data_subset['PID'].to_numpy(), data_subset['RID'].to_numpy())
    model_wb.censoring_limit = 0.0005
    model_wb.fit(method='mle-fast', global_search=True)
    # model_ln = models.Model_Trilinear_LogNormal()
    # model_ln.add_data(data_subset['PID'].to_numpy(), data_subset['RID'].to_numpy())
    # model_ln.censoring_limit = 0.005
    # model_ln.fit(method='mle', global_search=False)

    pidmax = 0.05

    model_pid = np.linspace(0.00, pidmax, 1000)
    model_rid_50_wb = model_wb.evaluate_inverse_cdf(0.50, model_pid)
    model_rid_20_wb = model_wb.evaluate_inverse_cdf(0.20, model_pid)
    model_rid_80_wb = model_wb.evaluate_inverse_cdf(0.80, model_pid)

    # model_rid_50_ln = model_ln.evaluate_inverse_cdf(0.50, model_pid)
    # model_rid_20_ln = model_ln.evaluate_inverse_cdf(0.20, model_pid)
    # model_rid_80_ln = model_ln.evaluate_inverse_cdf(0.80, model_pid)

    roll = models.Model()
    roll.add_data(data_subset['PID'].to_numpy(), data_subset['RID'].to_numpy())
    roll.calculate_rolling_quantiles()

    scatter_kwargs = {
        's': 2.2,
        'facecolor': 'none',
        'edgecolor': 'black',
        'alpha': 0.25,
    }

    roll_pid = roll.rolling_pid[roll.rolling_pid < pidmax]
    roll_rid_50 = roll.rolling_rid_50[roll.rolling_pid < pidmax]
    roll_rid_20 = roll.rolling_rid_20[roll.rolling_pid < pidmax]
    roll_rid_80 = roll.rolling_rid_80[roll.rolling_pid < pidmax]

    # Residual drift fragility curve (for probabilities)
    rid_capacity_delta = 0.01
    rid_capacity_beta = 0.3

    # Probability of excessive drift
    pid_conditioning_values = np.arange(0.0025, pidmax, 0.00125)

    halfwidth = 0.004
    window_loc = 0.025

    prob_empirical = []
    prob_pelicun_model_wb = []
    # prob_pelicun_model_ln = []

    num_realizations = 1000
    for val in pid_conditioning_values:
        subset_pairs = data_subset
        subset_pairs = subset_pairs[subset_pairs['PID'] > val - halfwidth]
        subset_pairs = subset_pairs[subset_pairs['PID'] < val + halfwidth]
        if len(subset_pairs) < 20:  # noqa: PLR2004
            prob_empirical.append(np.nan)
            prob_pelicun_model_wb.append(np.nan)
            # prob_pelicun_model_ln.append(np.nan)
        else:
            pid_array = np.random.choice(
                subset_pairs['PID'].values, size=num_realizations, replace=True
            )
            capacities = rid_capacity_delta * np.exp(
                rid_capacity_beta * norm.rvs(size=num_realizations)
            )
            rids_empirical = np.random.choice(
                subset_pairs['RID'].values, size=num_realizations, replace=True
            )
            rids_pelicun_model_wb = model_wb.generate_rid_samples(pid_array)
            # rids_pelicun_model_ln = model_ln.generate_rid_samples(pid_array)
            prob_empirical.append(
                sum(rids_empirical > capacities) / float(num_realizations)
            )
            prob_pelicun_model_wb.append(
                sum(rids_pelicun_model_wb > capacities) / float(num_realizations)
            )
            # prob_pelicun_model_ln.append(
            #     sum(rids_pelicun_model_ln > capacities) / float(num_realizations)
            # )

    plt.close()
    custom_ticks = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    custom_text = ['0%', '1%', '2%', '3%', '4%', '5%']
    fig, axs = plt.subplots(
        2,
        2,
        sharey=False,
        sharex=False,
        figsize=(4.85, 3.5),
        gridspec_kw={
            'width_ratios': [2, 1],
            'height_ratios': [1, 2],
        },
    )
    axs[0, 0].plot(
        pid_conditioning_values,
        prob_empirical,
        color='tab:red',
    )
    axs[0, 0].plot(
        pid_conditioning_values,
        prob_pelicun_model_wb,
        color='C0',
        linestyle='dashed',
    )
    axs[0, 0].grid(which='both', alpha=0.30)
    axs[0, 0].set(ylim=(0, 1.00))
    axs[0, 0].set(ylabel='Probability of\nExcessive Drift')

    axs[1, 0].scatter(
        data_subset['PID'].to_numpy(),
        data_subset['RID'].to_numpy(),
        **scatter_kwargs,
        label='Simulation Data',
    )
    axs[1, 0].plot(
        roll_pid,
        roll_rid_50,
        color='tab:red',
        label='Empirical Sample',
    )
    # Simulation Data Rolling\nQuantiles: 20%, 50%, 80%
    axs[1, 0].plot(
        roll_pid,
        roll_rid_20,
        color='tab:red',
        linewidth=0.50,
    )
    axs[1, 0].plot(
        roll_pid,
        roll_rid_80,
        color='tab:red',
        linewidth=0.50,
    )
    axs[1, 0].plot(
        model_pid,
        model_rid_50_wb,
        color='C0',
        linestyle='dashed',
        label='Proposed Model',
        linewidth=1.5,
    )
    axs[1, 0].plot(
        model_pid,
        model_rid_20_wb,
        color='C0',
        linestyle='dashed',
    )
    axs[1, 0].plot(
        model_pid,
        model_rid_80_wb,
        color='C0',
        linestyle='dashed',
    )
    axs[1, 0].axvline(
        x=window_loc + halfwidth, color='black', linewidth=0.50, linestyle='dashdot'
    )
    axs[1, 0].axvline(
        x=window_loc - halfwidth, color='black', linewidth=0.50, linestyle='dashdot'
    )
    axs[0, 0].axvline(
        x=window_loc + halfwidth, color='black', linewidth=0.50, linestyle='dashdot'
    )
    axs[0, 0].axvline(
        x=window_loc - halfwidth, color='black', linewidth=0.50, linestyle='dashdot'
    )
    axs[1, 0].set(ylabel='RSDR')
    axs[1, 0].set(xlabel='PSDR')
    # axs[1, 0].text(
    #     0.05,
    #     0.90,
    #     'Quantiles: 20%, 50%, 80%',
    #     ha='left',
    #     transform=axs[1, 0].transAxes,
    # )
    c = 0.0
    axs[1, 0].grid(which='both', alpha=0.30)
    axs[1, 0].set_xticks(custom_ticks)
    axs[1, 0].set_xticklabels(custom_text)
    axs[0, 0].set_xticks(custom_ticks)
    axs[0, 0].set_xticklabels(custom_text)
    axs[1, 0].set_xticks(custom_ticks)
    axs[1, 0].set_xticklabels(custom_text)
    axs[1, 1].set_xticks(custom_ticks)
    axs[1, 1].set_xticklabels(custom_text)
    axs[1, 0].set_yticks(custom_ticks)
    axs[1, 0].set_yticklabels(custom_text)
    axs[1, 0].set(xlim=(-c, pidmax + c), ylim=(-c, 0.03 + c))
    axs[1, 0].axhline(
        y=model_wb.censoring_limit,
        color='black',
        linewidth=0.50,
        linestyle='dashdot',
    )
    axs[1, 1].axhline(
        y=model_wb.censoring_limit,
        color='black',
        linewidth=0.50,
        linestyle='dashdot',
    )
    subset_pairs = data_subset
    subset_pairs = subset_pairs[subset_pairs['PID'] > window_loc - halfwidth]
    subset_pairs = subset_pairs[subset_pairs['PID'] < window_loc + halfwidth]
    sns.ecdfplot(subset_pairs['RID'], ax=axs[1, 1], color='tab:red')
    # sns.ecdfplot(capacities, ax=axs[1, 1], color='C2', linestyle=':', label='Excessive Drift\nFragility Curve')
    axs[1, 1].set(xlabel='RSDR', ylabel='CDF')
    pid_array = np.full(1000, window_loc)
    sns.ecdfplot(
        model_wb.generate_rid_samples(pid_array),
        ax=axs[1, 1],
        color='C0',
        linestyle='dashed',
    )
    # sns.ecdfplot(model_ln.generate_rid_samples(pid_array), ax=axs[1, 1], color='C1', linestyle='dotted')
    axs[1, 1].set(xlim=(-c, 0.03 + c))
    axs[1, 1].grid(which='both', alpha=0.30)
    axs[0, 1].remove()
    fig.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    axs[1, 1].text(0.45, 0.45, '(c)', transform=axs[1, 1].transAxes)
    axs[1, 0].text(0.30, 0.45, '(a)', transform=axs[1, 0].transAxes)
    axs[0, 0].text(0.30, 0.45, '(b)', transform=axs[0, 0].transAxes)
    fig.savefig(f'{figpath}/figure_matplotlib.pdf')
    fig.savefig(f'{figpath}/figure_matplotlib.svg')
