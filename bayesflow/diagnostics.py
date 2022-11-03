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

from scipy.stats import binom
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import logging
logging.basicConfig()

from bayesflow.computational_utilities import expected_calibration_error, simultaneous_ecdf_bands
from bayesflow.helper_classes import LossHistory
from bayesflow.helper_functions import check_posterior_prior_shapes


def plot_recovery(post_samples, prior_samples, point_agg=np.mean, uncertainty_agg=np.std, 
                  param_names=None, fig_size=None, label_fontsize=14, title_fontsize=16,
                  metric_fontsize=16, add_corr=True, add_r2=True, color='#8f2727', 
                  n_col=None, n_row=None):
    
    """ Creates and plots publication-ready recovery plot with true vs. point estimate + uncertainty.
    The point estimate can be controlled with the `point_agg` argument, and the uncertainty estimate
    can be controlled with the `uncertainty_agg` argument.

    This plot yields the same information as the "posterior z-score":

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
    point_agg         : callable, optional, default: np.mean
        The function to apply to the posterior draws to get a point estimate for each marginal.
    uncertainty_agg   : callable or None, optional, default: np.std
        The function to apply to the posterior draws to get an uncertainty estimate.
        If `None` provided, a simple scatter will be plotted.
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
    add_corr          : boolean, optional, default: True
        A flag for adding correlation between true and estimates to the plot.
    add_r2            : boolean, optional, default: True
        A flag for adding R^2 between true and estimates to the plot.
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and errobars.
        
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
    
    # Compute point estimates and uncertainties
    est = point_agg(post_samples, axis=1)
    if uncertainty_agg is not None:
        u = uncertainty_agg(post_samples, axis=1)
    
    # Determine n params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f'$p_{i}$' for i in range(1, n_params+1)]
        
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

    for i, ax in enumerate(axarr.flat):
        if i >= n_params:
            break

        # Add scatter and errorbars
        if uncertainty_agg is not None:
            im = ax.errorbar(prior_samples[:, i], est[:, i], yerr=u[:, i], fmt='o', alpha=0.5, color=color)
        else:
            im = ax.scatter(prior_samples[:, i], est[:, i], alpha=0.5, color=color)

        # Make plots quadratic to avoid visual illusions
        lower = min(prior_samples[:, i].min(), est[:, i].min())
        upper = max(prior_samples[:, i].max(), est[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps]) 
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_ylim()[0], ax.get_ylim()[1]], 
                 color='black', alpha=0.9, linestyle='dashed')
        
        # Add labels, optional metrics and title
        ax.set_xlabel('Ground truth', fontsize=label_fontsize)
        ax.set_ylabel('Estimated', fontsize=label_fontsize)
        if add_r2:
            r2 = r2_score(prior_samples[:, i], est[:, i])
            ax.text(0.1, 0.9, '$R^2$ = {:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=ax.transAxes, 
                     size=metric_fontsize)
        if add_corr:
            corr = np.corrcoef(prior_samples[:, i], est[:, i])[0, 1]
            ax.text(0.1, 0.8, '$r$ = {:.3f}'.format(corr),
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax.transAxes, 
                         size=metric_fontsize)
        ax.set_title(param_names[i], fontsize=title_fontsize)
        
        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
    f.tight_layout()
    return f


def plot_sbc_ecdf(post_samples, prior_samples, difference=False, stacked=False, fig_size=None, 
                  param_names=None, label_fontsize=14, legend_fontsize=14, title_fontsize=16, 
                  rank_ecdf_color='#a34f4f', fill_color='grey', **kwargs):
    """ Creates the empirical CDFs for each marginal rank distribution and plots it against
    a uniform ECDF. ECDF simultaneous bands are drawn using simulations from the uniform. Inspired by:

    https://arxiv.org/abs/2103.10522
    
    This figure is useful for models with many parameters and is supposed to give an idea
    of the overall calibration of a posterior approximator.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    difference        : boolean, optional, default: False
        If `True`, plots the ECDF difference. Enables a more dynamic visualization range.
    stacked           : boolean, optional, default: False
        If `True`, all ECDFs will be plotted on the same plot. If `False`, each ECDF will
        have its own subplot, similar to the behavior of `plot_sbc_histograms`.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None. Only relevant if `stacked=False`.
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 14
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text
    title_fontsize    : int, optional, default: 16
        The font size of the title text. Only relevant if `stacked=False`
    rank_ecdf_color   : str, optional, default: '#a34f4f'
        The color to use for the rank ECDFs
    fill_color        : str, optional, default: 'grey'
        The color of the fill arguments.
    **kwargs          : dict, optional, default: {}
        Keyword arguments can be passed to control the behavior of ECDF simultaneous band computation
        through the `ecdf_bands_kwargs` dictionary. See `simultaneous_ecdf_bands` for keyword arguments

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
        f, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        # Determine n_subplots dynamically
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))

        # Determine fig_size dynamically, if None
        if fig_size is None:
            fig_size = (int(5*n_col), int(5*n_row))
        
        # Initialize figure
        f, ax = plt.subplots(n_row, n_col, figsize=fig_size)
    
    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1]+1)/float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx
        
        if stacked:
            if j == 0:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label='Rank ECDFs')
            else:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        else: 
            ax.flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label='Rank ECDF')
       
    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(post_samples.shape[0], **kwargs.pop('ecdf_bands_kwargs', {}))

    # Difference, if specified
    if difference:
        L -= z
        H -= z

    # Add simultaneous bounds
    if stacked:
        titles = [None]
        axes = [ax]

    else:
        axes = ax.flat
        if param_names is None:
            titles = [f'$p_{i}$' for i in range(1, n_params+1)]
        else:
            titles = param_names

    for _ax, title in zip(axes, titles):
        _ax.fill_between(z, L, H, color=fill_color, alpha=0.2, label=fr'{int((1-alpha) * 100)}\% Confidence Bands')
        _ax.plot(z, L, color='black', alpha=0.3)
        _ax.plot(z, H, color='black', alpha=0.3)
        
        # Prettify plot
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)
        _ax.legend(fontsize=legend_fontsize)
        _ax.set_xlabel('Fractional rank statistic', fontsize=label_fontsize)
        if difference:
            ylab = 'ECDF difference'
        else:
            ylab = 'ECDF'
        _ax.set_ylabel(ylab, fontsize=label_fontsize)
        _ax.set_title(title, fontsize=title_fontsize)

    f.tight_layout()
    return f


def plot_sbc_histograms(post_samples, prior_samples, param_names=None, fig_size=None, num_bins=None, 
             binomial_interval=0.99, label_fontsize=14, title_fontsize=16, hist_color='#a34f4f'):
    """ Creates and plots publication-ready histograms of rank statistics for simulation-based calibration 
    (SBC) checks according to:

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
        logger.info(f'The ratio of simulations / posterior draws should be > 20 ' + 
                    f'for reliable variance reduction, but your ratio is {ratio}.\
                    Confidence intervals might be unreliable!')

    # Set n_bins automatically, if nothing provided
    if num_bins is None:
        num_bins = int(ratio / 2)
        # Attempt a fix if a single bin is determined so plot still makes sense
        if num_bins == 1:
            num_bins = 5

    # Determine n params and param names if None given
    if param_names is None:
        param_names = [f'$p_{i}$' for i in range(1, n_params+1)]
        
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
    mean = N / num_bins # corresponds to binom.mean(N, 1 / num_bins)

    # Plot marginal histograms in a loop
    if n_row > 1:
        ax = axarr.flat
    else:
        ax = axarr
    for j in range(len(param_names)):
        ax[j].axhspan(endpoints[0], endpoints[1], facecolor='gray', alpha=0.3)
        ax[j].axhline(mean, color='gray', zorder=0, alpha=0.5)
        sns.histplot(ranks[:, j], kde=False, ax=ax[j], color=hist_color, bins=num_bins, alpha=0.95)
        ax[j].set_title(param_names[j], fontsize=title_fontsize)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].set_xlabel('Rank statistic', fontsize=label_fontsize)
        ax[j].get_yaxis().set_ticks([])
        ax[j].set_ylabel('')
    f.tight_layout()
    return f


def plot_posterior_2d(posterior_draws, prior=None, prior_draws=None, param_names=None, height=2, 
                      legend_fontsize=14, post_color='#8f2727', prior_color='gray', post_alpha=0.9, 
                      prior_alpha=0.7):
    """ Generates a bivariate pairplot given posterior draws and prior.

    posterior_draws   : np.ndarray of shape (n_post_draws, n_params)
        The posterior draws obtained for a SINGLE observed data set.
    prior             : bayesflow.forward_inference.Prior instance or None, optional (default: None)
        The optional prior object having an input-output signature as given by ayesflow.forward_inference.Prior
    prior_draws       : np.ndarray of shape (n_prior_draws, n_params) or None, optonal (default: None)
        The optional prior draws obtained from the prior. If both prior and prior_draws are provided, prior_draws
        will be used.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    height            : float, optional, default: 2.
        The height of the pairplot.
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text.
    post_color        : str, optional, default: '#8f2727'
        The color for the posterior histograms and KDEs.
    priors_color      : str, optional, default: gray
        The color for the optional prior histograms and KDEs.
    post_alpha        : float in [0, 1], optonal, default: 0.9
        The opacity of the posterior plots.
    prior_alpha       : float in [0, 1], optonal, default: 0.7
        The opacity of the prior plots.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the shape of posterior_draws is not 2-dimensional.
    """
    
    # Ensure correct shape
    assert(len(posterior_draws.shape)) == 2, 'Shape of `posterior_samples` for a single data set should be 2 dimensional!'
    
    # Obtain n_draws and n_params
    n_draws, n_params = posterior_draws.shape

    # If prior object is given and no draws, obtain draws
    if prior is not None and prior_draws is None:
        draws = prior(n_draws)
        if type(draws) is dict:
            prior_draws = draws['prior_draws']
        else:
            prior_draws = draws
    # Otherwise, keep as is (prior_draws either filled or None)
    else:
        pass
    
    # Attempt to determine parameter names
    if param_names is None:
        if hasattr(prior, 'param_names'):
            param_names = prior.param_names
        else:
            param_names = [f'Param. {p}' for p in range(1, n_params+1)]

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
        handles = [Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
                   Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha)]
        g.fig.legend(handles, ['Posterior', 'Prior'], fontsize=legend_fontsize, loc='center right')

    # Remove upper axis
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].axis('off')

    # Add grids
    for i in range(n_params):
        for j in range(n_params):
            g.axes[i, j].grid(alpha=0.5)
    
    g.tight_layout()
    return g.fig


def plot_losses(history, fig_size=None, color='#8f2727', label_fontsize=14, title_fontsize=16):
    """ A generic helper function to plot the losses of a series of training runs.
    
    Parameters
    ----------
    
    history : pd.DataFrame or bayesflow.LossHistory object
        The (plottable) history as returned by a train_[...] method of a `Trainer` instance.
        
    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    # Handle non-pd.DataFrame type
    if type(history) is LossHistory:
        history = history.get_plottable()
    
    # Determine the number of rows for plot
    n_row = len(history.columns)
    
    # Initialize figure
    if fig_size is None:
        fig_size = (16, int(4 * n_row))
    f, axarr = plt.subplots(n_row, 1, figsize=fig_size)
    
    # Get the number of steps as an array
    step_index = np.arange(1, len(history)+1)
    
    # Loop through loss entries and populate plot
    looper = [axarr] if n_row == 1 else axarr.flat
    for i, ax in enumerate(looper):
        ax.plot(step_index, history.iloc[:, i], color=color, lw=2)
        ax.set_xlabel('Training step #', fontsize=label_fontsize)
        ax.set_ylabel('Loss value', fontsize=label_fontsize)
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.set_title(history.columns[i], fontsize=title_fontsize)
    f.tight_layout()
    return f


def plot_prior2d(prior, param_names=None, n_samples=2000, height=2.5, color='#8f2727', **kwargs):
    """ Creates pairplots for a given joint prior.
    
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
        prior_samples = prior_samples['prior_draws']

    # Get latent dimensionality and prepare titles
    dim = prior_samples.shape[-1]

    # Convert samples to a pandas data frame
    if param_names is None:
        titles = [f'Prior Param. {i}' for i in range(1, dim+1)]
    else:
        titles = [f'Prior {p}' for p in param_names]
    data_to_plot = pd.DataFrame(prior_samples, columns=titles)

    # Generate plots
    g = sns.PairGrid(data_to_plot, height=height, **kwargs)
    g.map_diag(sns.histplot, fill=True, color=color, alpha=0.9, kde=True)
    # Kernel density estimation (KDE) may not always be possible (e.g. with parameters whose correlation is close to 1 or -1).
    # In this scenario, a scatter-plot is generated instead.
    try:
        g.map_lower(sns.kdeplot, fill=True, color=color, alpha=0.9)
    except Exception as e:
        logging.warn("KDE failed due to the following exception:\n"+repr(e)+"\nSubstituting scatter plot.")
        g.map_lower(plt.scatter, alpha=0.6, s=40, edgecolor='k', color=color)
    g.map_upper(plt.scatter, alpha=0.6, s=40, edgecolor='k', color=color)
    
    # Add grids
    for i in range(dim):
        for j in range(dim):
            g.axes[i, j].grid(alpha=0.5)
    g.tight_layout()
    return g.fig


def plot_latent_space_2d(z_samples, height=2.5, color='#8f2727', **kwargs):
    """ Creates pairplots for the latent space learned by the inference network. Enables
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
    titles = [f'Latent Dim. {i}' for i in range(1, z_dim+1)]
    data_to_plot = pd.DataFrame(z_samples, columns=titles)

    # Generate plots
    g = sns.PairGrid(data_to_plot, height=height, **kwargs)
    g.map_diag(sns.histplot, fill=True, color=color, alpha=0.9, kde=True)
    g.map_lower(sns.kdeplot, fill=True, color=color, alpha=0.9)
    g.map_upper(plt.scatter, alpha=0.6, s=40, edgecolor='k', color=color)

    # Add grids
    for i in range(z_dim):
        for j in range(z_dim):
            g.axes[i, j].grid(alpha=0.5)
    g.tight_layout()
    return g.fig


def plot_calibration_curves(m_true, m_pred, model_names=None, n_bins=10, font_size=12, fig_size=(12, 4)):
    """ Plots the calibration curves and the ECE for a model comparison problem. Depends on the
    `expected_calibration_error` function for computing the ECE.


    Parameters
    ----------
    TODO
    """

    n_models = m_pred.shape[-1]
    if model_names is None:
        model_names = [f'$M_{m}$' for m in range(1, n_models+1)]

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
