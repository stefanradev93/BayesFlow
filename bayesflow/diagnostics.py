# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import binom, median_abs_deviation
from sklearn.metrics import confusion_matrix, r2_score

logging.basicConfig()

from bayesflow.computational_utilities import expected_calibration_error, simultaneous_ecdf_bands
from bayesflow.helper_functions import check_posterior_prior_shapes


def plot_recovery(
    post_samples,
    prior_samples,
    point_agg=np.median,
    uncertainty_agg=median_abs_deviation,
    param_names=None,
    fig_size=None,
    label_fontsize=16,
    title_fontsize=18,
    metric_fontsize=16,
    tick_fontsize=12,
    add_corr=True,
    add_r2=True,
    color="#8f2727",
    n_col=None,
    n_row=None,
):
    """Creates and plots publication-ready recovery plot with true vs. point estimate + uncertainty.
    The point estimate can be controlled with the ``point_agg`` argument, and the uncertainty estimate
    can be controlled with the ``uncertainty_agg`` argument.

    This plot yields similar information as the "posterior z-score", but allows for generic
    point and uncertainty estimates:

    https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html

    Important: Posterior aggregates play no special role in Bayesian inference and should only
    be used heuristically. For instanec, in the case of multi-modal posteriors, common point
    estimates, such as mean, (geometric) median, or maximum a posteriori (MAP) mean nothing.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws (true parameters) obtained for generating the n_data_sets
    point_agg         : callable, optional, default: np.median
        The function to apply to the posterior draws to get a point estimate for each marginal.
        The default computes the marginal median for each marginal posterior as a robust
        point estimate.
    uncertainty_agg   : callable or None, optional, default: scipy.stats.median_abs_deviation
        The function to apply to the posterior draws to get an uncertainty estimate.
        If ``None`` provided, a simple scatter using only ``point_agg`` will be plotted.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    add_corr          : bool, optional, default: True
        A flag for adding correlation between true and estimates to the plot
    add_r2            : bool, optional, default: True
        A flag for adding R^2 between true and estimates to the plot
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and errobars

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    # Sanity check
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Compute point estimates and uncertainties
    est = point_agg(post_samples, axis=1)
    if uncertainty_agg is not None:
        u = uncertainty_agg(post_samples, axis=1)

    # Determine n params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    # turn axarr into 1D list
    if n_col > 1 or n_row > 1:
        axarr_it = axarr.flat
    else:
        # for 1x1, axarr is not a list -> turn it into one for use with enumerate
        axarr_it = [axarr]

    for i, ax in enumerate(axarr_it):
        if i >= n_params:
            break

        # Add scatter and errorbars
        if uncertainty_agg is not None:
            _ = ax.errorbar(prior_samples[:, i], est[:, i], yerr=u[:, i], fmt="o", alpha=0.5, color=color)
        else:
            _ = ax.scatter(prior_samples[:, i], est[:, i], alpha=0.5, color=color)

        # Make plots quadratic to avoid visual illusions
        lower = min(prior_samples[:, i].min(), est[:, i].min())
        upper = max(prior_samples[:, i].max(), est[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps])
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )

        # Add optional metrics and title
        if add_r2:
            r2 = r2_score(prior_samples[:, i], est[:, i])
            ax.text(
                0.1,
                0.9,
                "$R^2$ = {:.3f}".format(r2),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        if add_corr:
            corr = np.corrcoef(prior_samples[:, i], est[:, i])[0, 1]
            ax.text(
                0.1,
                0.8,
                "$r$ = {:.3f}".format(corr),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        ax.set_title(param_names[i], fontsize=title_fontsize)

        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Ground truth", fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axarr[0].set_ylabel("Estimated", fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for _ax in axarr[:, 0]:
            _ax.set_ylabel("Estimated", fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr_it[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f


def plot_z_score_contraction(
    post_samples,
    prior_samples,
    param_names=None,
    fig_size=None,
    label_fontsize=16,
    title_fontsize=18,
    tick_fontsize=12,
    color="#8f2727",
    n_col=None,
    n_row=None,
):
    """Implements a graphical check for global model sensitivity by plotting the posterior
    z-score over the posterior contraction for each set of posterior samples in ``post_samples``
    according to [1].

    - The definition of the posterior z-score is:

    post_z_score = (posterior_mean - true_parameters) / posterior_std

    And the score is adequate if it centers around zero and spreads roughly in the interval [-3, 3]

    - The definition of posterior contraction is:

    post_contraction = 1 - (posterior_variance / prior_variance)

    In other words, the posterior is a proxy for the reduction in ucnertainty gained by
    replacing the prior with the posterior. The ideal posterior contraction tends to 1.
    Contraction near zero indicates that the posterior variance is almost identical to
    the prior variance for the particular marginal parameter distribution.

    Note: Means and variances will be estimated vie their sample-based estimators.

    [1] Schad, D. J., Betancourt, M., & Vasishth, S. (2021).
    Toward a principled Bayesian workflow in cognitive science.
    Psychological methods, 26(1), 103.

    Also available at https://arxiv.org/abs/1904.12765

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws (true parameters) obtained for generating the n_data_sets
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and errobars

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    # Sanity check for shape integrity
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Estimate posterior means and stds
    post_means = post_samples.mean(axis=1)
    post_stds = post_samples.std(axis=1, ddof=1)
    post_vars = post_samples.var(axis=1, ddof=1)

    # Estimate prior variance
    prior_vars = prior_samples.var(axis=0, keepdims=True, ddof=1)

    # Compute contraction
    post_cont = 1 - (post_vars / prior_vars)

    # Compute posterior z score
    z_score = (post_means - prior_samples) / post_stds

    # Determine number of params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    # turn axarr into 1D list
    if n_col > 1 or n_row > 1:
        axarr_it = axarr.flat
    else:
        # for 1x1, axarr is not a list -> turn it into one for use with enumerate
        axarr_it = [axarr]

    # Loop and plot
    for i, ax in enumerate(axarr_it):
        if i >= n_params:
            break

        ax.scatter(post_cont[:, i], z_score[:, i], color=color, alpha=0.5)
        ax.set_title(param_names[i], fontsize=title_fontsize)
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
        ax.set_xlim([-0.05, 1.05])

    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Posterior contraction", fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axarr[0].set_ylabel("Posterior z-score", fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for _ax in axarr[:, 0]:
            _ax.set_ylabel("Posterior z-score", fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr_it[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f


def plot_sbc_ecdf(
    post_samples,
    prior_samples,
    difference=False,
    stacked=False,
    fig_size=None,
    param_names=None,
    label_fontsize=16,
    legend_fontsize=14,
    title_fontsize=18,
    tick_fontsize=12,
    rank_ecdf_color="#a34f4f",
    fill_color="grey",
    **kwargs,
):
    """Creates the empirical CDFs for each marginal rank distribution and plots it against
    a uniform ECDF. ECDF simultaneous bands are drawn using simulations from the uniform,
    as proposed by [1].

    For models with many parameters, use `stacked=True` to obtain an idea of the overall calibration
    of a posterior approximator.

    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test for discrete uniformity and
    its applications in goodness-of-fit evaluation and multiple sample comparison. Statistics and Computing,
    32(2), 1-21. https://arxiv.org/abs/2103.10522

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    difference        : bool, optional, default: False
        If `True`, plots the ECDF difference. Enables a more dynamic visualization range.
    stacked           : bool, optional, default: False
        If `True`, all ECDFs will be plotted on the same plot. If `False`, each ECDF will
        have its own subplot, similar to the behavior of `plot_sbc_histograms`.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None. Only relevant if `stacked=False`.
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text
    title_fontsize    : int, optional, default: 16
        The font size of the title text. Only relevant if `stacked=False`
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    rank_ecdf_color   : str, optional, default: '#a34f4f'
        The color to use for the rank ECDFs
    fill_color        : str, optional, default: 'grey'
        The color of the fill arguments.
    **kwargs          : dict, optional, default: {}
        Keyword arguments can be passed to control the behavior of ECDF simultaneous band computation
        through the ``ecdf_bands_kwargs`` dictionary. See `simultaneous_ecdf_bands` for keyword arguments

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

    # Sanity checks
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Store reference to number of parameters
    n_params = post_samples.shape[-1]

    # Compute fractional ranks (using broadcasting)
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1) / post_samples.shape[1]

    # Prepare figure
    if stacked:
        n_row, n_col = 1, 1
        f, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        # Determine n_subplots dynamically
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))

        # Determine fig_size dynamically, if None
        if fig_size is None:
            fig_size = (int(5 * n_col), int(5 * n_row))

        # Initialize figure
        f, ax = plt.subplots(n_row, n_col, figsize=fig_size)

    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        if stacked:
            if j == 0:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDFs")
            else:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        else:
            ax.flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDF")

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    # Difference, if specified
    if difference:
        L -= z
        H -= z
        ylab = "ECDF difference"
    else:
        ylab = "ECDF"

    # Add simultaneous bounds
    if stacked:
        titles = [None]
        axes = [ax]
    else:
        axes = ax.flat
        if param_names is None:
            titles = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            titles = param_names

    for _ax, title in zip(axes, titles):
        _ax.fill_between(z, L, H, color=fill_color, alpha=0.2, label=rf"{int((1-alpha) * 100)}$\%$ Confidence Bands")

        # Prettify plot
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)
        _ax.legend(fontsize=legend_fontsize)
        _ax.set_title(title, fontsize=title_fontsize)
        _ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        _ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    if stacked:
        bottom_row = [ax]
    else:
        bottom_row = ax if n_row == 1 else ax[-1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Fractional rank statistic", fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axes[0].set_ylabel(ylab, fontsize=label_fontsize)
    else:  # if there is more than one row, the ax array is 2D
        for _ax in ax[:, 0]:
            _ax.set_ylabel(ylab, fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axes[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f


def plot_sbc_histograms(
    post_samples,
    prior_samples,
    param_names=None,
    fig_size=None,
    num_bins=None,
    binomial_interval=0.99,
    label_fontsize=16,
    title_fontsize=18,
    tick_fontsize=12,
    hist_color="#a34f4f",
):
    """Creates and plots publication-ready histograms of rank statistics for simulation-based calibration
    (SBC) checks according to [1].

    Any deviation from uniformity indicates miscalibration and thus poor convergence
    of the networks or poor combination between generative model / networks.

    [1] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
    Validating Bayesian inference algorithms with simulation-based calibration.
    arXiv preprint arXiv:1804.06788.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None
    num_bins          : int, optional, default: 10
        The number of bins to use for each marginal histogram
    binomial_interval : float in (0, 1), optional, default: 0.95
        The width of the confidence interval for the binomial distribution
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    hist_color        : str, optional, default '#a34f4f'
        The color to use for the histogram body

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

    # Sanity check
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Determine the ratio of simulations to prior draws
    n_sim, n_draws, n_params = post_samples.shape
    ratio = int(n_sim / n_draws)

    # Log a warning if N/B ratio recommended by Talts et al. (2018) < 20
    if ratio < 20:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info(
            f"The ratio of simulations / posterior draws should be > 20 "
            + f"for reliable variance reduction, but your ratio is {ratio}.\
                    Confidence intervals might be unreliable!"
        )

    # Set n_bins automatically, if nothing provided
    if num_bins is None:
        num_bins = int(ratio / 2)
        # Attempt a fix if a single bin is determined so plot still makes sense
        if num_bins == 1:
            num_bins = 5

    # Determine n params and param names if None given
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine n_subplots dynamically
    n_row = int(np.ceil(n_params / 6))
    n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = (int(5 * n_col), int(5 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

    # Compute ranks (using broadcasting)
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1)

    # Compute confidence interval and mean
    N = int(prior_samples.shape[0])
    # uniform distribution expected -> for all bins: equal probability
    # p = 1 / num_bins that a rank lands in that bin
    endpoints = binom.interval(binomial_interval, N, 1 / num_bins)
    mean = N / num_bins  # corresponds to binom.mean(N, 1 / num_bins)

    # Plot marginal histograms in a loop
    if n_row > 1:
        ax = axarr.flat
    else:
        ax = axarr
    for j in range(len(param_names)):
        ax[j].axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.3)
        ax[j].axhline(mean, color="gray", zorder=0, alpha=0.9)
        sns.histplot(ranks[:, j], kde=False, ax=ax[j], color=hist_color, bins=num_bins, alpha=0.95)
        ax[j].set_title(param_names[j], fontsize=title_fontsize)
        ax[j].spines["right"].set_visible(False)
        ax[j].spines["top"].set_visible(False)
        ax[j].get_yaxis().set_ticks([])
        ax[j].set_ylabel("")
        ax[j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax[j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Rank statistic", fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f


def plot_posterior_2d(
    posterior_draws,
    prior=None,
    prior_draws=None,
    param_names=None,
    height=3,
    label_fontsize=14,
    legend_fontsize=16,
    tick_fontsize=12,
    post_color="#8f2727",
    prior_color="gray",
    post_alpha=0.9,
    prior_alpha=0.7,
):
    """Generates a bivariate pairplot given posterior draws and optional prior or prior draws.

    posterior_draws   : np.ndarray of shape (n_post_draws, n_params)
        The posterior draws obtained for a SINGLE observed data set.
    prior             : bayesflow.forward_inference.Prior instance or None, optional, default: None
        The optional prior object having an input-output signature as given by ayesflow.forward_inference.Prior
    prior_draws       : np.ndarray of shape (n_prior_draws, n_params) or None, optonal (default: None)
        The optional prior draws obtained from the prior. If both prior and prior_draws are provided, prior_draws
        will be used.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    height            : float, optional, default: 3
        The height of the pairplot
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    legend_fontsize   : int, optional, default: 16
        The font size of the legend text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    post_color        : str, optional, default: '#8f2727'
        The color for the posterior histograms and KDEs
    priors_color      : str, optional, default: gray
        The color for the optional prior histograms and KDEs
    post_alpha        : float in [0, 1], optonal, default: 0.9
        The opacity of the posterior plots
    prior_alpha       : float in [0, 1], optonal, default: 0.7
        The opacity of the prior plots

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the shape of posterior_draws is not 2-dimensional.
    """

    # Ensure correct shape
    assert (
        len(posterior_draws.shape)
    ) == 2, "Shape of `posterior_samples` for a single data set should be 2 dimensional!"

    # Obtain n_draws and n_params
    n_draws, n_params = posterior_draws.shape

    # If prior object is given and no draws, obtain draws
    if prior is not None and prior_draws is None:
        draws = prior(n_draws)
        if type(draws) is dict:
            prior_draws = draws["prior_draws"]
        else:
            prior_draws = draws
    # Otherwise, keep as is (prior_draws either filled or None)
    else:
        pass

    # Attempt to determine parameter names
    if param_names is None:
        if hasattr(prior, "param_names"):
            if prior.param_names is not None:
                param_names = prior.param_names
            else:
                param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Pack posterior draws into a dataframe
    posterior_draws_df = pd.DataFrame(posterior_draws, columns=param_names)

    # Add posterior
    g = sns.PairGrid(posterior_draws_df, height=height)
    g.map_diag(sns.histplot, fill=True, color=post_color, alpha=post_alpha, kde=True)
    g.map_lower(sns.kdeplot, fill=True, color=post_color, alpha=post_alpha)

    # Add prior, if given
    if prior_draws is not None:
        prior_draws_df = pd.DataFrame(prior_draws, columns=param_names)
        g.data = prior_draws_df
        g.map_diag(sns.histplot, fill=True, color=prior_color, alpha=prior_alpha, kde=True, zorder=-1)
        g.map_lower(sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1)

    # Add legend, if prior also given
    if prior_draws is not None or prior is not None:
        handles = [
            Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
            Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha),
        ]
        g.fig.legend(handles, ["Posterior", "Prior"], fontsize=legend_fontsize, loc="center right")

    # Remove upper axis
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].axis("off")

    # Modify tick sizes
    for i, j in zip(*np.tril_indices_from(g.axes, 1)):
        g.axes[i, j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        g.axes[i, j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Add nice labels
    for i, param_name in enumerate(param_names):
        g.axes[i, 0].set_ylabel(param_name, fontsize=label_fontsize)
        g.axes[len(param_names) - 1, i].set_xlabel(param_name, fontsize=label_fontsize)

    # Add grids
    for i in range(n_params):
        for j in range(n_params):
            g.axes[i, j].grid(alpha=0.5)

    g.tight_layout()
    return g.fig


def plot_losses(
    train_losses,
    val_losses=None,
    fig_size=None,
    train_color="#8f2727",
    val_color="black",
    lw_train=2,
    lw_val=3,
    grid_alpha=0.5,
    legend_fontsize=14,
    label_fontsize=14,
    title_fontsize=16,
):
    """A generic helper function to plot the losses of a series of training epochs and runs.

    Parameters
    ----------

    train_losses      : pd.DataFrame
        The (plottable) history as returned by a train_[...] method of a ``Trainer`` instance.
        Alternatively, you can just pass a data frame of validation losses instead of train losses,
        if you only want to plot the validation loss.
    val_losses        : pd.DataFrame or None, optional, default: None
        The (plottable) validation history as returned by a train_[...] method of a ``Trainer`` instance.
        If left ``None``, only train losses are plotted. Should have the same number of columns
        as ``train_losses``.
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor. Inferred if ``None``
    train_color       : str, optional, default: '#8f2727'
        The color for the train loss trajectory
    val_color         : str, optional, default: black
        The color for the optional validation loss trajectory
    lw_train          : int, optional, default: 2
        The linewidth for the training loss curve
    lw_val            : int, optional, default: 3
        The linewidth for the validation loss curve
    grid_alpha        : float, optional, default 0.5
        The opacity factor for the background gridlines
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the number of columns in ``train_losses`` does not match the
        number of columns in ``val_losses``.
    """

    # Determine the number of rows for plot
    n_row = len(train_losses.columns)

    # Initialize figure
    if fig_size is None:
        fig_size = (16, int(4 * n_row))
    f, axarr = plt.subplots(n_row, 1, figsize=fig_size)

    # Get the number of steps as an array
    train_step_index = np.arange(1, len(train_losses) + 1)
    if val_losses is not None:
        val_step = int(np.floor(len(train_losses) / len(val_losses)))
        val_step_index = train_step_index[(val_step - 1) :: val_step]

        # If unequal length due to some reason, attempt a fix
        if val_step_index.shape[0] > val_losses.shape[0]:
            val_step_index = val_step_index[: val_losses.shape[0]]

    # Loop through loss entries and populate plot
    looper = [axarr] if n_row == 1 else axarr.flat
    for i, ax in enumerate(looper):
        # Plot train curve
        ax.plot(train_step_index, train_losses.iloc[:, i], color=train_color, lw=lw_train, alpha=0.9, label="Training")
        # Plot optional val curve
        if val_losses is not None:
            if i < val_losses.shape[1]:
                ax.plot(
                    val_step_index,
                    val_losses.iloc[:, i],
                    linestyle="--",
                    marker="o",
                    color=val_color,
                    lw=lw_val,
                    label="Validation",
                )
            # Schmuck
        ax.set_xlabel("Training step #", fontsize=label_fontsize)
        ax.set_ylabel("Loss value", fontsize=label_fontsize)
        sns.despine(ax=ax)
        ax.grid(alpha=grid_alpha)
        ax.set_title(train_losses.columns[i], fontsize=title_fontsize)
        # Only add legend if there is a validation curve
        if val_losses is not None:
            ax.legend(fontsize=legend_fontsize)
    f.tight_layout()
    return f


def plot_prior2d(prior, param_names=None, n_samples=2000, height=2.5, color="#8f2727", **kwargs):
    """Creates pairplots for a given joint prior.

    Parameters
    ----------
    prior       : callable
        The prior object which takes a single integer argument and generates random draws.
    param_names : list of str or None, optional, default None
        An optional list of strings which
    n_samples   : int, optional, default: 1000
        The number of random draws from the joint prior
    height      : float, optional, default: 2.5
        The height of the pair plot
    color       : str, optional, defailt : '#8f2727'
        The color of the plot
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    # Generate prior draws
    prior_samples = prior(n_samples)

    # Handle dict type
    if type(prior_samples) is dict:
        prior_samples = prior_samples["prior_draws"]

    # Get latent dimensionality and prepare titles
    dim = prior_samples.shape[-1]

    # Convert samples to a pandas data frame
    if param_names is None:
        titles = [f"Prior Param. {i}" for i in range(1, dim + 1)]
    else:
        titles = [f"Prior {p}" for p in param_names]
    data_to_plot = pd.DataFrame(prior_samples, columns=titles)

    # Generate plots
    g = sns.PairGrid(data_to_plot, height=height, **kwargs)
    g.map_diag(sns.histplot, fill=True, color=color, alpha=0.9, kde=True)
    # Kernel density estimation (KDE) may not always be possible (e.g. with parameters whose correlation is close to 1 or -1).
    # In this scenario, a scatter-plot is generated instead.
    try:
        g.map_lower(sns.kdeplot, fill=True, color=color, alpha=0.9)
    except Exception as e:
        logging.warn("KDE failed due to the following exception:\n" + repr(e) + "\nSubstituting scatter plot.")
        g.map_lower(plt.scatter, alpha=0.6, s=40, edgecolor="k", color=color)
    g.map_upper(plt.scatter, alpha=0.6, s=40, edgecolor="k", color=color)

    # Add grids
    for i in range(dim):
        for j in range(dim):
            g.axes[i, j].grid(alpha=0.5)
    g.tight_layout()
    return g.fig


def plot_latent_space_2d(z_samples, height=2.5, color="#8f2727", **kwargs):
    """Creates pairplots for the latent space learned by the inference network. Enables
    visual inspection of the the latent space and whether its structrue corresponds to the
    one enforced by the optimization criterion.

    Parameters
    ----------
    z_samples   : np.ndarray or tf.Tensor of shape (n_sim, n_params)
        The latent samples computed through a forward pass of the inference network.
    height      : float, optional, default: 2.5
        The height of the pair plot.
    color       : str, optional, defailt : '#8f2727'
        The color of the plot
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    # Try to convert z_samples, if eventually tf.Tensor is passed
    if type(z_samples) is not np.ndarray:
        z_samples = z_samples.numpy()

    # Get latent dimensionality and prepare titles
    z_dim = z_samples.shape[-1]

    # Convert samples to a pandas data frame
    titles = [f"Latent Dim. {i}" for i in range(1, z_dim + 1)]
    data_to_plot = pd.DataFrame(z_samples, columns=titles)

    # Generate plots
    g = sns.PairGrid(data_to_plot, height=height, **kwargs)
    g.map_diag(sns.histplot, fill=True, color=color, alpha=0.9, kde=True)
    g.map_lower(sns.kdeplot, fill=True, color=color, alpha=0.9)
    g.map_upper(plt.scatter, alpha=0.6, s=40, edgecolor="k", color=color)

    # Add grids
    for i in range(z_dim):
        for j in range(z_dim):
            g.axes[i, j].grid(alpha=0.5)
    g.tight_layout()
    return g.fig


def plot_calibration_curves(
    true_models,
    pred_models,
    model_names=None,
    num_bins=10,
    label_fontsize=16,
    legend_fontsize=14,
    title_fontsize=18,
    tick_fontsize=12,
    fig_size=None,
    color="#8f2727",
):
    """Plots the calibration curves, the ECEs and the marginal histograms of predicted posterior model probabilities
    for a model comparison problem. The marginal histograms inform about the fraction of predictions in each bin.
    Depends on the ``expected_calibration_error`` function for computing the ECE.

    Parameters
    ----------
    true_models       : np.ndarray of shape (num_data_sets, num_models)
        The one-hot-encoded true model indices per data set.
    pred_models       : np.ndarray of shape (num_data_sets, num_models)
        The predicted posterior model probabilities (PMPs) per data set.
    model_names       : list or None, optional, default: None
        The model names for nice plot titles. Inferred if None.
    num_bins          : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text (ECE value)
    title_fontsize    : int, optional, default: 16
        The font size of the title text. Only relevant if `stacked=False`
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor. Inferred if ``None``
    color             : str, optional, default: '#8f2727'
        The color of the calibration curves

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving
    """

    num_models = true_models.shape[-1]
    if model_names is None:
        model_names = [rf"$M_{{{m}}}$" for m in range(1, num_models + 1)]

    # Determine n_subplots dynamically
    n_row = int(np.ceil(num_models / 6))
    n_col = int(np.ceil(num_models / n_row))
    cal_errs, cal_probs = expected_calibration_error(true_models, pred_models, num_bins)

    # Initialize figure
    if fig_size is None:
        fig_size = (int(5 * n_col), int(5 * n_row))
    fig, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    if n_row > 1:
        ax = axarr.flat

    # Plot marginal calibration curves in a loop
    if n_row > 1:
        ax = axarr.flat
    else:
        ax = axarr
    for j in range(num_models):
        # Plot calibration curve
        ax[j].plot(cal_probs[j][0], cal_probs[j][1], color=color)

        # Plot AB line
        ax[j].plot(ax[j].get_xlim(), ax[j].get_xlim(), "--", color="darkgrey")

        # Plot PMP distribution over bins
        uniform_bins = np.linspace(0.0, 1.0, num_bins + 1)
        norm_weights = np.ones_like(pred_models) / len(pred_models)
        ax[j].hist(pred_models[:, j], bins=uniform_bins, weights=norm_weights[:, j], color="grey", alpha=0.3)

        # Tweak plot
        ax[j].spines["right"].set_visible(False)
        ax[j].spines["top"].set_visible(False)
        ax[j].set_xlim([0, 1])
        ax[j].set_ylim([0, 1])
        ax[j].set_xlabel("Predicted probability", fontsize=label_fontsize)
        ax[j].set_ylabel("True probability", fontsize=label_fontsize)
        ax[j].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].grid(alpha=0.5)
        ax[j].text(
            0.1,
            0.9,
            r"$\widehat{{\mathrm{{ECE}}}}$ = {0:.3f}".format(cal_errs[j]),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax[j].transAxes,
            size=legend_fontsize,
        )
        ax[j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax[j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

        # Set title
        ax[j].set_title(model_names[j], fontsize=title_fontsize)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    true_models,
    pred_models,
    model_names=None,
    fig_size=(5, 5),
    title_fontsize=18,
    tick_fontsize=12,
    xtick_rotation=None,
    ytick_rotation=None,
    normalize=True,
    cmap=None,
    title=True,
):
    """Plots a confusion matrix for validating a neural network trained for Bayesian model comparison.

    Parameters
    ----------
    true_models    : np.ndarray of shape (num_data_sets, num_models)
        The one-hot-encoded true model indices per data set.
    pred_models    : np.ndarray of shape (num_data_sets, num_models)
        The predicted posterior model probabilities (PMPs) per data set.
    model_names    : list or None, optional, default: None
        The model names for nice plot titles. Inferred if None.
    fig_size       : tuple or None, optional, default: (5, 5)
        The figure size passed to the ``matplotlib`` constructor. Inferred if ``None``
    title_fontsize : int, optional, default: 18
        The font size of the title text.
    tick_fontsize  : int, optional, default: 12
        The font size of the axis label and model name texts.
    xtick_rotation: int, optional, default: None
        Rotation of x-axis tick labels (helps with long model names).
    ytick_rotation: int, optional, default: None
        Rotation of y-axis tick labels (helps with long model names).
    normalize      : bool, optional, default: True
        A flag for normalization of the confusion matrix.
        If True, each row of the confusion matrix is normalized to sum to 1.
    cmap           : matplotlib.colors.Colormap or str, optional, default: None
        Colormap to be used for the cells. If a str, it should be the name of a registered colormap,
        e.g., 'viridis'. Default colormap matches the BayesFlow defaults by ranging from white to red.
    title          : bool, optional, default True
        A flag for adding 'Confusion Matrix' above the matrix.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving
    """

    if model_names is None:
        num_models = true_models.shape[-1]
        model_names = [rf"$M_{{{m}}}$" for m in range(1, num_models + 1)]

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("", ["white", "#8f2727"])

    # Flatten input
    true_models = np.argmax(true_models, axis=1)
    pred_models = np.argmax(pred_models, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(true_models, pred_models)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink=0.7)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(model_names, fontsize=tick_fontsize)
    if xtick_rotation:
       plt.xticks(rotation=xtick_rotation, ha="right")
    ax.set_yticklabels(model_names, fontsize=tick_fontsize)
    if ytick_rotation:
       plt.yticks(rotation=ytick_rotation)
    ax.set_xlabel("Predicted model", fontsize=tick_fontsize)
    ax.set_ylabel("True model", fontsize=tick_fontsize)

    # Loop over data dimensions and create text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black"
            )
    if title:
        ax.set_title("Confusion Matrix", fontsize=title_fontsize)
    return fig


def plot_mmd_hypothesis_test(mmd_null,
                             mmd_observed=None,
                             alpha_level=0.05,
                             null_color=(0.16407, 0.020171, 0.577478),
                             observed_color="red",
                             alpha_color="orange",
                             truncate_vlines_at_kde=False,
                             xmin=None,
                             xmax=None,
                             bw_factor=1.5):
    """

    Parameters
    ----------
    mmd_null: np.ndarray
        samples from the MMD sampling distribution under the null hypothesis "the model is well-specified"
    mmd_observed: float
        observed MMD value
    alpha_level: float
        rejection probability (type I error)
    null_color: color
        color for the H0 sampling distribution
    observed_color: color
        color for the observed MMD
    alpha_color: color
        color for the rejection area
    truncate_vlines_at_kde: bool
        true: cut off the vlines at the kde
        false: continue kde lines across the plot
    xmin: float
        lower x axis limit
    xmax: float
        upper x axis limit
    bw_factor: float, default: 1.5
        bandwidth (aka. smoothing parameter) of the kernel density estimate

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    """

    def draw_vline_to_kde(x, kde_object, color, label=None, **kwargs):
        kde_x, kde_y = kde_object.lines[0].get_data()
        idx = np.argmin(np.abs(kde_x - x))
        plt.vlines(x=x, ymin=0, ymax=kde_y[idx], color=color, linewidth=3, label=label, **kwargs)

    def fill_area_under_kde(kde_object, x_start, x_end=None, **kwargs):
        kde_x, kde_y = kde_object.lines[0].get_data()
        if x_end is not None:
            plt.fill_between(kde_x, kde_y, where=(kde_x >= x_start) & (kde_x <= x_end),
                             interpolate=True, **kwargs)
        else:
            plt.fill_between(kde_x, kde_y, where=(kde_x >= x_start),
                             interpolate=True, **kwargs)

    f = plt.figure(figsize=(8, 4))

    kde = sns.kdeplot(mmd_null, fill=False, linewidth=0, bw_adjust=bw_factor)
    sns.kdeplot(mmd_null, fill=True, alpha=.12, color=null_color, bw_adjust=bw_factor)

    if truncate_vlines_at_kde:
        draw_vline_to_kde(x=mmd_observed, kde_object=kde, color=observed_color, label=r"Observed data")
    else:
        plt.vlines(x=mmd_observed, ymin=0, ymax=plt.gca().get_ylim()[1], color=observed_color, linewidth=3,
                   label=r"Observed data")

    mmd_critical = np.quantile(mmd_null, 1 - alpha_level)
    fill_area_under_kde(kde, mmd_critical, color=alpha_color, alpha=0.5, label=fr"{int(alpha_level*100)}% rejection area")

    if truncate_vlines_at_kde:
        draw_vline_to_kde(x=mmd_critical, kde_object=kde, color=alpha_color)
    else:
        plt.vlines(x=mmd_critical, color=alpha_color, linewidth=3, ymin=0, ymax=plt.gca().get_ylim()[1])

    sns.kdeplot(mmd_null, fill=False, linewidth=3, color=null_color, label=r"$H_0$", bw_adjust=bw_factor)

    plt.xlabel(r"MMD", fontsize=20)
    plt.ylabel("")
    plt.yticks([])
    plt.xlim(xmin, xmax)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.legend(fontsize=20)
    sns.despine()

    return f
