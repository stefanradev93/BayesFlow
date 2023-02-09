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

# Corresponds to Task T.10 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np
from scipy.integrate import odeint

bayesflow_benchmark_info = {
    "simulator_is_batched": False,
    "parameter_names": [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$"],
    "configurator_info": "posterior",
}


def prior(rng=None):
    """Generates a random draw from a 4-dimensional (independent) lognormal prior
    which represents the four contact parameters of the Lotka-Volterra model.

    Parameters
    ----------
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    theta : np.ndarray of shape (4,)
        A single draw from the 4-dimensional prior.
    """

    if rng is None:
        rng = np.random.default_rng()

    theta = rng.lognormal(mean=[-0.125, -3, -0.125, -3], sigma=0.5)
    return theta


def _deriv(x, t, alpha, beta, gamma, delta):
    """Helper function for scipy.integrate.odeint."""

    X, Y = x
    dX = alpha * X - beta * X * Y
    dY = -gamma * Y + delta * X * Y
    return dX, dY


def simulator(theta, X0=30, Y0=1, T=20, subsample=10, flatten=True, obs_noise=0.1, rng=None):
    """Runs a Lotka-Volterra simulation for T time steps and returns `subsample` evenly spaced
    points from the simulated trajectory, given contact parameters `theta`.

    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.10.

    Parameters
    ----------
    theta       : np.ndarray of shape (2,)
        The 2-dimensional vector of disease parameters.
    X0          : float, optional, default: 30
        Initial number of prey species.
    Y0          : float, optional, default: 1
        Initial number of predator species.
    T           : T, optional, default: 20
        The duration (time horizon) of the simulation.
    subsample   : int or None, optional, default: 10
        The number of evenly spaced time points to return. If None,
        no subsampling will be performed and all T timepoints will be returned.
    flatten     : bool, optional, default: True
        A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
        representation of the simulated data is returned.
    obs_noise   : float, optional, default: 0.1
        The standard deviation of the log-normal likelihood.
    rng         : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (subsample, 2) or (subsample*2,) if `subsample is not None`,
        otherwise shape (T, 2) or (T*2,) if `subsample is None`.
        The time series of simulated predator and pray populations
    """

    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Create vector (list) of initial conditions
    x0 = X0, Y0

    # Unpack parameter vector into scalars
    alpha, beta, gamma, delta = theta

    # Prepate time vector between 0 and T of length T
    t_vec = np.linspace(0, T, T)

    # Integrate using scipy and retain only infected (2-nd dimension)
    pp = odeint(_deriv, x0, t_vec, args=(alpha, beta, gamma, delta))

    # Subsample evenly the specified number of points, if specified
    if subsample is not None:
        pp = pp[:: (T // subsample)]

    # Ensure minimum count is 0, which will later pass by log(0 + 1)
    pp[pp < 0] = 0.0

    # Add noise, decide whether to flatten and return
    x = rng.lognormal(np.log1p(pp), sigma=obs_noise)
    if flatten:
        return x.flatten()
    return x


def configurator(forward_dict, mode="posterior", scale_data=1000, as_summary_condition=False):
    """Configures simulator outputs for use in BayesFlow training."""

    # Case only posterior configuration
    if mode == "posterior":
        input_dict = _config_posterior(forward_dict, scale_data, as_summary_condition)

    # Case only likelihood configuration
    elif mode == "likelihood":
        input_dict = _config_likelihood(forward_dict, scale_data)

    # Case posterior and likelihood configuration
    elif mode == "joint":
        input_dict = {}
        input_dict["posterior_inputs"] = _config_posterior(forward_dict, scale_data, as_summary_condition)
        input_dict["likelihood_inputs"] = _config_likelihood(forward_dict, scale_data)

    # Throw otherwise
    else:
        raise NotImplementedError('For now, only a choice between ["posterior", "likelihood", "joint"] is available!')
    return input_dict


def _config_posterior(forward_dict, scale_data, as_summary_condition):
    """Helper function for posterior configuration."""

    input_dict = {}
    input_dict["parameters"] = forward_dict["prior_draws"].astype(np.float32)
    if as_summary_condition:
        input_dict["summary_conditions"] = forward_dict["sim_data"].astype(np.float32) / scale_data
    else:
        input_dict["direct_conditions"] = forward_dict["sim_data"].astype(np.float32) / scale_data
    return input_dict


def _config_likelihood(forward_dict, scale_data):
    """Helper function for likelihood configuration."""

    input_dict = {}
    input_dict["observables"] = forward_dict["sim_data"].astype(np.float32) / scale_data
    input_dict["conditions"] = forward_dict["prior_draws"].astype(np.float32)
    return input_dict
