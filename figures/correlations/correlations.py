"""
Code to generate figures for the journal paper.
Title: Residual correlations and proposed model
Written: Mon Sep 30 05:09:56 PM PDT 2024
By Ioannis Vouvakis Manousakis of UC Berkeley.
"""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm

from src import models
from src.handle_data import load_dataset, remove_collapse


def estimate_corr(x_i: float, x_j: float, params: tuple[float, float]) -> float:
    c_1, c_2 = params
    return np.exp(-c_1 * np.abs(x_i - x_j) ** c_2)


seed_value = 32
np.random.seed(seed_value)
output_path = Path('figures/correlations')
if not output_path.exists():
    output_path.mkdir(parents=True)

# Retrieve all data
all_data = remove_collapse(
    load_dataset(Path('extra/structural_analysis/results/data.parquet'))
)

results = {}
# for system, stories, rc in tqdm(
#     list(
#         product(
#             ('smrf', 'scbf', 'brbf'),
#             ('3', '6', '9'),
#             ('ii', 'iv'),
#         )
#     )
# ):
#     # limit results to particular archetype
#     data = (
#         all_data.loc[(system, stories, rc), ['PID', 'RID']].dropna(how='all').abs()
#     )
#     num_stories = int(stories)
#     # hazard levels of interest
#     if rc == 'ii':
#         hazard_levels = ['20', '21', '22']
#     else:
#         hazard_levels = ['23', '24', '25']
#     data_hz = data.loc[(hazard_levels,), :]
#     # empirical max RID of all stories
#     empirical_max_rid = data_hz['RID'].unstack('loc').max(axis=1)
#     # fit a max-max model
#     data_maxmax = data.groupby(['hz', 'gm', 'dr']).max()
#     model_maxmax = models.Model_Trilinear_Weibull()
#     model_maxmax.add_data(
#         data_maxmax['PID'].to_numpy(), data_maxmax['RID'].to_numpy()
#     )
#     model_maxmax.censoring_limit = 0.00025
#     model_maxmax.fit(method='mle-fast', global_search=True)
#     # fit story-by-story models
#     story_models = {}
#     for story in data.index.get_level_values('loc').unique():
#         model_story = models.Model_Trilinear_Weibull()
#         data_story = data[data.index.get_level_values('loc') == story]
#         model_story.add_data(
#             data_story['PID'].to_numpy(), data_story['RID'].to_numpy()
#         )
#         model_story.censoring_limit = 0.00025
#         model_story.fit(method='mle-fast', global_search=True)
#         story_models[story] = model_story
#     # # fit a multivariate lognormal to the PIDs and generate samples
#     # pids = data.loc[(hazard_levels,), 'PID'].unstack('loc')
#     # log_pids = np.log(pids)
#     # mean = log_pids.mean().to_numpy()
#     # covariance = np.cov(log_pids.T)
#     # multivariate_dist = multivariate_normal(mean=mean, cov=covariance)
#     num_realizations = 10000
#     # pid_sample = pd.DataFrame(
#     #     np.exp(multivariate_dist.rvs(size=num_realizations)), columns=pids.columns
#     # )
#     # don't fit a multivariate lognormal to the PIDs: just bootstrap samples
#     pids = data.loc[(hazard_levels,), 'PID'].unstack('loc')
#     bootstrap_sample = np.random.choice(range(len(pids)), size=num_realizations)
#     pid_sample = pd.DataFrame(
#         pids.to_numpy()[bootstrap_sample, :], columns=pids.columns
#     )
#     # obtain correlation structure from RID data
#     rids = data.loc[(hazard_levels,), 'RID'].unstack('loc')
#     copula_correlations = np.empty(shape=(num_stories, num_stories))
#     copula_correlations_default = np.empty(shape=(num_stories, num_stories))
#     for i_story, j_story in combinations(
#         data.index.get_level_values('loc').unique(), 2
#     ):
#         rid_i = rids[i_story]
#         pid_i = pids[i_story]
#         rid_j = rids[j_story]
#         pid_j = pids[j_story]
#         # calculate residuals and capture residual correlation
#         median_i = story_models[i_story].evaluate_inverse_cdf(0.50, pid_i)
#         median_j = story_models[j_story].evaluate_inverse_cdf(0.50, pid_j)
#         residual_i = rid_i - median_i
#         residual_j = rid_j - median_j
#         coeff = np.corrcoef(residual_i, residual_j)[0, 1]
#         copula_correlations[int(i_story) - 1, int(j_story) - 1] = coeff
#         copula_correlations[int(j_story) - 1, int(i_story) - 1] = coeff
#         # # Old approach - Capturing the RID correlations without
#         # # accounting for PID correlation:
#         # # transform to uniform sample
#         # u_i = story_models[i_story].evaluate_cdf(rid_i, pid_i)
#         # u_j = story_models[j_story].evaluate_cdf(rid_j, pid_j)
#         # # transform to normal sample
#         # z_i = norm.ppf(u_i)
#         # z_j = norm.ppf(u_j)
#         # # capture correlation
#         # coeff = np.corrcoef(z_i, z_j)[0, 1]
#         # copula_correlations[int(i_story) - 1, int(j_story) - 1] = coeff
#         # copula_correlations[int(j_story) - 1, int(i_story) - 1] = coeff
#         # copula_correlations based on "Default" correlation model
#         story_heights = [15.00]
#         for _ in range(num_stories - 1):
#             story_heights.append(story_heights[-1] + 13.00)
#         story_heights = np.array(story_heights)
#         story_heights /= 15.0 + 13.0 * (num_stories - 1)
#         x_i = story_heights[int(i_story) - 1]
#         x_j = story_heights[int(j_story) - 1]
#         estimate = estimate_corr(x_i, x_j, (10.00, 2.00))
#         copula_correlations_default[int(i_story) - 1, int(j_story) - 1] = estimate
#         copula_correlations_default[int(j_story) - 1, int(i_story) - 1] = estimate
#     for i_story in range(num_stories):
#         copula_correlations[i_story, i_story] = 1.00
#         copula_correlations_default[i_story, i_story] = 1.00
#     l_mat = np.linalg.cholesky(copula_correlations)
#     l_mat_default = np.linalg.cholesky(copula_correlations_default)
#     story_rids = {}
#     story_rids_default = {}
#     # sample uniforms
#     u_mat = np.random.uniform(0.00, 1.00, size=(num_realizations, num_stories))
#     # transform to normal
#     z_mat = norm.ppf(u_mat)
#     # Apply the Cholesky transformation to obtain correlated normal
#     # realizations
#     z_mat_corr = (l_mat @ z_mat.T).T
#     z_mat_corr_default = (l_mat_default @ z_mat.T).T
#     # Back to uniform
#     u_mat_corr = norm.cdf(z_mat_corr)
#     u_mat_corr_default = norm.cdf(z_mat_corr_default)
#     for story in data.index.get_level_values('loc').unique():
#         model_story = story_models[story]
#         model_story.uniform_sample = u_mat_corr[:, int(story) - 1]
#         # model_story.uniform_sample = None
#         story_rids[story] = model_story.generate_rid_samples(
#             pid_sample[story].to_numpy()
#         )
#         model_story.uniform_sample = u_mat_corr_default[:, int(story) - 1]
#         # model_story.uniform_sample = None
#         story_rids_default[story] = model_story.generate_rid_samples(
#             pid_sample[story].to_numpy()
#         )
#         story_rids_df = pd.DataFrame(story_rids)
#         story_rids_df_default = pd.DataFrame(story_rids_default)
#         max_rid_from_stories = story_rids_df.max(axis=1)
#         max_rid_from_stories_default = story_rids_df_default.max(axis=1)
#     rid_maxmax = model_maxmax.generate_rid_samples(pid_sample.max(axis=1))
#     results[system, stories, rc] = {
#         'story_models': story_models,
#         'model_maxmax': model_maxmax,
#         'values': {
#             'empirical_max_rid': empirical_max_rid,
#             'max_rid_from_stories': max_rid_from_stories,
#             'max_rid_from_stories_default': max_rid_from_stories_default,
#             'rid_maxmax': rid_maxmax,
#             'copula_correlations': copula_correlations,
#             'copula_correlations_default': copula_correlations_default,
#         },
#     }


# with Path(f'{output_path}/results.pcl').open('wb') as f:
#     dill.dump(results, f)

with Path(f'{output_path}/results.pcl').open('rb') as f:
    results = dill.load(f)


plt.close()
num_rows = 3
num_columns = 7
fig, axs = plt.subplots(num_rows, num_columns, figsize=(4.85, 4.85))
i = 0
j = -1
for stories, system, rc in tqdm(
    list(
        product(
            ('3', '6', '9'),
            ('smrf', 'scbf', 'brbf'),
            ('ii', 'iv'),
        )
    )
):
    j += 1
    if j == num_columns - 1:
        j = 0
        i += 1
    ax = axs[i, j]
    data = results[(system, stories, rc)]['values']['copula_correlations']
    ax.imshow(data, cmap='BuGn', interpolation='nearest', vmin=0.00, vmax=1.00)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set(title='-'.join([system.upper(), stories.upper(), rc.upper()]))
for i, stories in enumerate(('3', '6', '9')):
    ax = axs[i, num_columns - 1]
    data = results[('smrf', stories, 'ii')]['values']['copula_correlations_default']
    ax.imshow(data, cmap='BuGn', interpolation='nearest', vmin=0.00, vmax=1.00)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout(pad=1.00)
plt.savefig(f'{output_path}/figure_matplotlib.svg')
plt.show()
