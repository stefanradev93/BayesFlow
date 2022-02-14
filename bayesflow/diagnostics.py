# Copyright 2022 The BayesFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bayesflow.computational_utilities import expected_calibration_error


def plot_sbc(post_samples, prior_samples, param_names=None, fig_size=None, 
             num_bins=10, binomial_interval=0.95, label_fontsize=14, title_fontsize=16):

    """ Creates and plots publication-ready histograms for simulation-based calibration 
    checks according to:

    Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). 
    Validating Bayesian inference algorithms with simulation-based calibration. 
    arXiv preprint arXiv:1804.06788.

    Any deviation from uniformity indicates miscalibration and thus poor convergence 
    of the networks or poor combination between generative model / networks.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    num_bins          : int, optional, default: 10
        The number of bins to use for each marginal histogram
    binomial_interval : float in (0, 1), optional, default: 0.95
        The width of the confidence interval for the binomial distribution
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """
    
    # Determine n params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f'p_{i}' for i in range(1, n_params+1)]
        
    # Determine n_subplots dynamically
    n_row = int(np.ceil(n_params / 6))
    n_col = int(np.ceil(n_params / n_row))
    
    # Initialize figure
    if fig_size is None:
        fig_size = (20, int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

    # Compute ranks (using broadcasting)    
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1)

    # Compute confidence interval
    N = int(prior_samples.shape[0])
    endpoints = binom.interval(binomial_interval, N, 1 / (num_bins+1))

    # Plot marginal histograms in a loop
    if n_row > 1:
        ax = axarr.flat
    else:
        ax = axarr
    for j in range(len(param_names)):

        ax[j].axhspan(endpoints[0], endpoints[1], facecolor='gray', alpha=0.3)
        ax[j].axhline(np.mean(endpoints), color='gray', zorder=0, alpha=0.5)
        sns.histplot(ranks[:, j], kde=False, ax=ax[j], color='#a34f4f', bins=num_bins, alpha=0.95)
        ax[j].set_title(param_names[j], fontsize=title_fontsize)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].set_xlabel('Rank statistic', fontsize=label_fontsize)
        ax[j].get_yaxis().set_ticks([])
        ax[j].set_ylabel('')
    f.tight_layout()
    return f

def plot_calibration_curves(m_true, m_pred, model_names=None, n_bins=10, font_size=12, fig_size=(12, 4)):
    """Plots the calibration curves for a model comparison problem.

    Parameters
    ----------
    TODO
    """

    n_models = m_pred.shape[-1]
    if model_names is None:
        model_names = [f'M_{m}' for m in range(1, n_models+1)]

    # Determine n_subplots dynamically
    n_row = int(np.ceil(n_models / 6))
    n_col = int(np.ceil(n_models / n_row))

    cal_errs, cal_probs = expected_calibration_error(m_true, m_pred, n_bins)

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    if n_row > 1:
        ax = axarr.flat

    # Plot marginal calibration curves in a loop
    if n_row > 1:
        ax = axarr.flat
    else:
        ax = axarr
    for j in range(n_models):

        # Plot calibration curve
        ax[j].plot(cal_probs[j][0], cal_probs[j][1])

        # Plot AB line
        ax[j].plot(ax[j].get_xlim(), ax[j].get_xlim(), '--', color='black')

        # Tweak plot
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].set_xlim([0, 1])
        ax[j].set_ylim([0, 1])
        ax[j].set_xlabel('Accuracy')
        ax[j].set_ylabel('Confidence')
        ax[j].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].text(0.1, 0.9, r'$\widehat{{ECE}}$ = {0:.3f}'.format(cal_errs[j]),
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax[j].transAxes,
                        size=font_size)

        # Set title
        ax.set_title(model_names[j])
    f.tight_layout()
    return f