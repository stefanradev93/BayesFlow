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

# Corresponds to Task T.9 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np
from scipy.integrate import odeint

bayesflow_benchmark_info = {
    "simulator_is_batched": False,
    "parameter_names": [r"$\beta$", r"$\gamma$"],
    "configurator_info": "posterior",
}


def prior(rng=None):
    """Generates a random draw from a 2-dimensional (independent) lognormal prior
    which represents the contact and recovery rate parameters of a basic SIR model.

    Parameters
    ----------
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    theta : np.ndarray of shape (2,)
        A single draw from the 2-dimensional prior.
    """

    if rng is None:
        rng = np.random.default_rng()

    theta = rng.lognormal(mean=[np.log(0.4), np.log(1 / 8)], sigma=[0.5, 0.2])
    return theta


def _deriv(x, t, N, beta, gamma):
    """Helper function for scipy.integrate.odeint."""

    S, I, R = x
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return dS, dI, dR


def simulator(theta, N=1e6, T=160, I0=1.0, R0=0.0, subsample=10, total_count=1000, scale_by_total=True, rng=None):
    """Runs a basic SIR model simulation for T time steps and returns `subsample` evenly spaced
    points from the simulated trajectory, given disease parameters (contact and recovery rate) `theta`.

    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.9.

    Note, that the simulator will scale the outputs between 0 and 1.

    Parameters
    ----------
    theta          : np.ndarray of shape (2,)
        The 2-dimensional vector of disease parameters.
    N              : float, optional, default: 1e6 = 1 000 000
        The size of the simulated population.
    T              : T, optional, default: 160
        The duration (time horizon) of the simulation.
    I0             : float, optional, default: 1.
        The number of initially infected individuals.
    R0             : float, optional, default: 0.
        The number of initially recovered individuals.
    subsample      : int or None, optional, default: 10
        The number of evenly spaced time points to return. If None,
        no subsampling will be performed and all T timepoints will be returned.
    total_count    : int, optional, default: 1000
        The N parameter of the binomial noise distribution. Used just
        for scaling the data and magnifying the effect of noise, such that
        max infected == total_count.
    scale_by_total : bool, optional, default: True
        Scales the outputs by ``total_count`` if set to True.
    rng            : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (subsample,) or (T,) if subsample=None
        The time series of simulated infected individuals. A trailing dimension of 1 should
        be added by a BayesFlow configurator if the data is (properly) to be treated as time series.
    """

    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Create vector (list) of initial conditions
    x0 = N - I0 - R0, I0, R0

    # Unpack parameter vector into scalars
    beta, gamma = theta

    # Prepate time vector between 0 and T of length T
    t_vec = np.linspace(0, T, T)

    # Integrate using scipy and retain only infected (2-nd dimension)
    irt = odeint(_deriv, x0, t_vec, args=(N, beta, gamma))[:, 1]

    # Subsample evenly the specified number of points, if specified
    if subsample is not None:
        irt = irt[:: (T // subsample)]

    # Truncate irt, so that small underflow below zero becomes zero
    irt = np.maximum(irt, 0.0)

    # Add noise and scale, if indicated
    x = rng.binomial(n=total_count, p=irt / N)
    if scale_by_total:
        x = x / total_count
    return x


def configurator(forward_dict, mode="posterior", as_summary_condition=False):
    """Configures simulator outputs for use in BayesFlow training."""

    # Case only posterior configuration
    if mode == "posterior":
        input_dict = _config_posterior(forward_dict, as_summary_condition)

    # Case only likelihood configuration
    elif mode == "likelihood":
        input_dict = _config_likelihood(forward_dict)

    # Case posterior and likelihood configuration
    elif mode == "joint":
        input_dict = {}
        input_dict["posterior_inputs"] = _config_posterior(forward_dict, as_summary_condition)
        input_dict["likelihood_inputs"] = _config_likelihood(forward_dict)

    # Throw otherwise
    else:
        raise NotImplementedError('For now, only a choice between ["posterior", "likelihood", "joint"] is available!')
    return input_dict


def _config_posterior(forward_dict, as_summary_condition):
    """Helper function for posterior configuration."""

    input_dict = {}
    input_dict["parameters"] = forward_dict["prior_draws"].astype(np.float32)
    if as_summary_condition:
        input_dict["summary_conditions"] = forward_dict["sim_data"].astype(np.float32)[:, :, np.newaxis]
    else:
        input_dict["direct_conditions"] = forward_dict["sim_data"].astype(np.float32)
    return input_dict


def _config_likelihood(forward_dict):
    """Helper function for likelihood configuration."""

    input_dict = {}
    input_dict["conditions"] = forward_dict["prior_draws"].astype(np.float32)
    input_dict["observables"] = forward_dict["sim_data"].astype(np.float32)
    return input_dict
