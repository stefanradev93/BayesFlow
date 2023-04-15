import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayesflow import computational_utilities


def model_misspecification_sensitivity(trainer, generator_misspecification, p1_config, p2_config,
                                       n_posterior_samples=500, n_sim=200):
    """

    Parameters
    ----------
    trainer: bayesflow.trainers.Trainer
        A ``Trainer`` instance (usually after converged training).
    generator_misspecification: callable with signature p1: float, p2, float -> bayesflow.simulation.GenerativeModel
        A callable that takes two (potentially misspecified) parameters and returns a generative model
        for forward sampling.
    p1_config: dict
        Configuration for the first potentially misspecified parameter ``p1``.
        fields: name (str), values (1D np.ndarray), well_specified_value (float)
    p2_config: dict
        Configuration for the second potentially misspecified parameter ``p2``.
        fields: name (str), values (1D np.ndarray), well_specified_value (float)
    n_posterior_samples: int
        number of samples from the approximate posterior per data set
    n_sim:
        number of simulated data sets per configuration

    Returns
    -------
    posterior_error_dict: {P1, P2, value} - dictionary with parameter grid (P1, P2) and posterior error results (value)
    summary_mmd: {P1, P2, value} - dictionary with parameter grid (P1, P2) and summary MMD results (value)

    """
    # setup the grid
    n1, n2 = len(p1_config["values"]), len(p2_config["values"])
    P2, P1 = np.meshgrid(p2_config["values"], p1_config["values"])

    posterior_error = np.zeros((n1, n2))
    summary_mmd = np.zeros((n1, n2))

    for i in tqdm(range(n1)):
        for j in range(n2):
            p1 = P1[i, j]
            p2 = P2[i, j]
            generative_model_ = generator_misspecification(p1, p2)
            simulations = trainer.configurator(generative_model_(n_sim))
            theta_true = simulations['parameters']

            theta_est = trainer.amortizer.sample(simulations, n_samples=n_posterior_samples)

            # RMSE computation
            posterior_error[i, j] = computational_utilities.aggregated_error(
                x_true=theta_true,
                x_pred=theta_est,
                inner_error_fun=computational_utilities.root_mean_squared_error,
                outer_aggregation_fun=np.mean
            )

            # MMD computation
            sim_trainer = trainer.configurator(trainer.generative_model(n_sim))
            s_trainer = trainer.amortizer.summary_net(sim_trainer['summary_conditions'])

            s_obs = trainer.amortizer.summary_net(simulations['summary_conditions'])

            summary_mmd[i, j] = np.sqrt(computational_utilities.maximum_mean_discrepancy(s_obs, s_trainer).numpy())

    # build output dictionaries
    posterior_error_dict = {"P1": P1, "P2": P2, "value": posterior_error, "name": "Posterior Error"}
    summary_mmd_dict = {"P1": P1, "P2": P2, "value": summary_mmd, "name": "Summary MMD"}

    return posterior_error_dict, summary_mmd_dict


def plot_grid(data, p1_config, p2_config, plot_config=None, type=""):
    """

    Parameters
    ----------
    data: dict, as output by :func:`bayesflow.sensitivity.model_misspecification_sensitivity`
    p1_config: dict
        see parameter `p1_config` in :func:`bayesflow.sensitivity.model_misspecification_sensitivity`
    p2_config: dict
        see parameter `p2_config` in :func:`bayesflow.sensitivity.model_misspecification_sensitivity`
    plot_config: dict
        plot configuration dictionary,
        fields: xticks, yticks, vmin, vmax, cmap, cbar_title
    type: str
        one of ["rmse", "mmd"], sets colorbar title and colorbar colormap

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    """
    if plot_config is None:
        plot_config = dict()

    # merge config dicts
    default_plot_config = {'xticks': None, 'yticks': None, 'vmin': 0, 'vmax': None, 'cmap': 'viridis'}

    if type.lower() == "rmse":
        default_plot_config["cmap"] = 'inferno'
        default_plot_config["cbar_title"] = "RMSE"

    elif type.lower() == "mmd":
        default_plot_config["cmap"] = 'viridis'
        default_plot_config["cbar_title"] = "MMD"

    plot_config = default_plot_config | plot_config

    # Construct plot
    fig = plt.figure(figsize=(10, 5))
    plt.pcolor(data['P1'], data['P2'], data['value'], shading="nearest", rasterized=True,
               cmap=plot_config['cmap'], vmin=plot_config['vmin'], vmax=plot_config['vmax'])
    plt.xlabel(p1_config["name"], fontsize=28)
    plt.ylabel(p2_config["name"], fontsize=28)

    plt.tick_params(labelsize=24)
    plt.axhline(y=p2_config["well_specified_value"], linestyle="--", color="lightgreen", alpha=.80)
    plt.axvline(x=p1_config["well_specified_value"], linestyle="--", color="lightgreen", alpha=.80)
    plt.xticks(plot_config['xticks'])
    plt.yticks(plot_config['yticks'])

    cbar = plt.colorbar(orientation="vertical")
    cbar.ax.set_ylabel(plot_config["cbar_title"], fontsize=20, labelpad=12)
    cbar.ax.tick_params(labelsize=20)

    return fig
