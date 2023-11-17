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


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import bayesflow.computational_utilities
from bayesflow import computational_utilities as utils


def misspecification_experiment(
    trainer,
    generator,
    first_config_dict,
    second_config_dict,
    error_function=bayesflow.computational_utilities.aggregated_error,
    n_posterior_samples=500,
    n_sim=200,
    configurator=None,
):
    """Performs a systematic sensitivity analysis with regard to two misspecification
    factors across different values of the factors provided in the config dictionaries.

    Parameters
    ----------
    trainer             : bayesflow.trainers.Trainer
        A ``Trainer`` instance (usually after converged training).
    generator           : callable with signature p1: float, p2, float -> ``simulation.GenerativeModel``
        A callable that takes two misspecification factors and returns a generative model
        for forward sampling responsible for generating n_sim simulations.
    first_config_dict   : dict
        Configuration for the first misspecification factor
        fields: name (str), values (1D np.ndarray)
    second_config_dict  : dict
        Configuration for the second misspecification factor
        fields: name (str), values (1D np.ndarray)
    error_function      : callable, default: bayesflow.computational_utilities.aggregated_error
        A callable that computes an error metric on the approximate posterior samples
    n_posterior_samples : int, optional, default: 500
        Number of samples from the approximate posterior per data set
    n_sim               : int, optional, default: 200
        Number of simulated data sets per configuration
    configurator        : callable or None, optional, default: None
        An optional configurator for the misspecified simulations.
        If ``None`` provided (default), ``Trainer.configurator`` will be used.

    Returns
    -------
    posterior_error_dict: {P1, P2, value} - dictionary with misspecification grid (P1, P2) and posterior error results (values)
    summary_mmd: {P1, P2, values} - dictionary with misspecification grid (P1, P2) and summary MMD results (values)
    """

    # Setup the grid and prepare placeholders
    n1, n2 = len(first_config_dict["values"]), len(second_config_dict["values"])
    P2, P1 = np.meshgrid(second_config_dict["values"], first_config_dict["values"])
    posterior_error = np.zeros((n1, n2))
    summary_mmd = np.zeros((n1, n2))

    for i in tqdm(range(n1)):
        for j in range(n2):
            # Create and configure simulations from misspecified model
            p1 = P1[i, j]
            p2 = P2[i, j]
            generative_model_ = generator(p1, p2)
            simulations = generative_model_(n_sim)
            if configurator is None:
                simulations = trainer.configurator(simulations)
            else:
                simulations = configurator(simulations)
            true_params = simulations["parameters"]
            param_samples = trainer.amortizer.sample(simulations, n_samples=n_posterior_samples)

            # RMSE computation
            posterior_error[i, j] = error_function(true_params, param_samples)

            # MMD computation
            sim_trainer = trainer.configurator(trainer.generative_model(n_sim))
            summary_well = trainer.amortizer.summary_net(sim_trainer["summary_conditions"])
            summary_miss = trainer.amortizer.summary_net(simulations["summary_conditions"])
            summary_mmd[i, j] = np.sqrt(utils.maximum_mean_discrepancy(summary_miss, summary_well).numpy())

    # Build output dictionaries
    posterior_error_dict = {"P1": P1, "P2": P2, "values": posterior_error, "name": "posterior_error"}
    summary_mmd_dict = {"P1": P1, "P2": P2, "values": summary_mmd, "name": "summary_mmd"}
    return posterior_error_dict, summary_mmd_dict


def plot_model_misspecification_sensitivity(results_dict, first_config_dict, second_config_dict, plot_config=None):
    """Visualizes the results from a sensitivity analysis via a colored 2D grid.

    Parameters
    ----------
    results_dict        : dict
        The results from :func:`sensitivity.misspecification_experiment`,
        Alternatively, a dictionary with mandatory keys: P1, P2, values
    first_config_dict   : dict
        see parameter `first_config_dict` in :func:`sensitivity.misspecification_experiment`
        Important: Needs additional key ``well_specified_value``
    second_config_dict  : dict
        see parameter `second_config_dict` in :func:`sensitivity.misspecification_experiment`
        Important: Needs additional key ``well_specified_value``
    plot_config         : dict or None, optional, default: None
        Optional plot configuration dictionary,
        fields: xticks, yticks, vmin, vmax, cmap, cbar_title

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    if plot_config is None:
        plot_config = dict()

    # merge config dicts
    default_plot_config = {
        "xticks": None,
        "yticks": None,
        "vmin": 0,
        "vmax": None,
        "cmap": "viridis",
        "cbar_title": None,
    }

    if results_dict["name"].lower() == "posterior_error":
        default_plot_config["cmap"] = "inferno"
        default_plot_config["cbar_title"] = "Posterior Error"

    elif results_dict["name"].lower() == "summary_mmd":
        default_plot_config["cmap"] = "viridis"
        default_plot_config["cbar_title"] = "Summary MMD"
    else:
        raise NotImplementedError("Only 'summary_mmd' or 'posterior_error' are currently supported as plot types!")

    plot_config = default_plot_config | plot_config

    f = plot_color_grid(
        x_grid=results_dict["P1"],
        y_grid=results_dict["P2"],
        z_grid=results_dict["values"],
        cmap=plot_config["cmap"],
        vmin=plot_config["vmin"],
        vmax=plot_config["vmax"],
        xlabel=first_config_dict["name"],
        ylabel=second_config_dict["name"],
        hline_location=second_config_dict["well_specified_value"],
        vline_location=first_config_dict["well_specified_value"],
        xticks=plot_config["xticks"],
        yticks=plot_config["yticks"],
        cbar_title=plot_config["cbar_title"],
    )
    return f


def plot_color_grid(
    x_grid,
    y_grid,
    z_grid,
    cmap="viridis",
    vmin=None,
    vmax=None,
    xlabel="x",
    ylabel="y",
    cbar_title="z",
    xticks=None,
    yticks=None,
    hline_location=None,
    vline_location=None,
):
    """Plots a 2-dimensional color grid.

    Parameters
    ----------
    x_grid         : np.ndarray
        meshgrid of x values
    y_grid         : np.ndarray
        meshgrid of y values
    z_grid         : np.ndarray
        meshgrid of z values (coded by color in the plot)
    cmap           : str, default: viridis
        color map for the fill
    vmin           : float, default: None
        lower limit of the color map, None results in dynamic limit
    vmax           : float, default: None
        upper limit of the color map, None results in dynamic limit
    xlabel         : str, default: x
        x label text
    ylabel         : str, default: y
        y label text
    cbar_title     : str, default: z
        title of the color bar legend
    xticks         : list, default: None
        list of x ticks, None results in dynamic ticks
    yticks         : list, default: None
        list of y ticks, None results in dynamic ticks
    hline_location : float, default: None
        (optional) horizontal dashed line
    vline_location : float, default: None
        (optional) vertical dashed line

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    # Construct plot
    fig = plt.figure(figsize=(10, 5))
    plt.pcolor(x_grid, y_grid, z_grid, shading="nearest", rasterized=True, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.tick_params(labelsize=24)

    if hline_location is not None:
        plt.axhline(y=hline_location, linestyle="--", color="lightgreen", alpha=0.80)
    if vline_location is not None:
        plt.axvline(x=vline_location, linestyle="--", color="lightgreen", alpha=0.80)

    plt.xticks(xticks)
    plt.yticks(yticks)

    cbar = plt.colorbar(orientation="vertical")
    cbar.ax.set_ylabel(cbar_title, fontsize=20, labelpad=12)
    cbar.ax.tick_params(labelsize=20)

    return fig
