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

# Corresponds to Task T.3 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np

bayesflow_benchmark_info = {
    "simulator_is_batched": False,
    "parameter_names": [r"$\theta_{}$".format(i) for i in range(1, 6)],
    "configurator_info": "posterior",
}


def prior(lower_bound=-3.0, upper_bound=3.0, rng=None):
    """Generates a random draw from a 5-dimensional uniform prior bounded between
    `lower_bound` and `upper_bound` which represents the 5 parameters of the SLCP
    simulator.

    Parameters
    ----------
    lower_bound : float, optional, default : -3
        The lower bound of the uniform prior.
    upper_bound : float, optional, default : 3
        The upper bound of the uniform prior.
    rng         : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    theta : np.ndarray of shape (5, )
        A single draw from the 5-dimensional uniform prior.
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low=lower_bound, high=upper_bound, size=5)


def simulator(theta, n_obs=4, flatten=True, rng=None):
    """Generates data from the SLCP model designed as a benchmark for a simple likelihood
    and a complex posterior due to a non-linear pushforward theta -> x.

    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.3

    Parameters
    ----------
    theta   : np.ndarray of shape (theta, D)
        The location parameters of the Gaussian likelihood.
    n_obs   : int, optional, default: 4
        The number of observations to generate from the slcp likelihood.
    flatten : bool, optional, default: True
        A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
        representation of the simulated data is returned.
    rng     : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (n_obs*2, ) or (n_obs, 2), as indictated by the `flatten`
        boolean flag. The sample of simulated data from the SLCP model.
    """

    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Specify 2D location
    loc = np.array([theta[0], theta[1]])

    # Specify 2D covariance matrix
    s1 = theta[2] ** 2
    s2 = theta[3] ** 2
    rho = np.tanh(theta[4])
    cov = rho * s1 * s2
    S_theta = np.array([[s1**2, cov], [cov, s2**2]])

    # Obtain given number of draws from the MVN likelihood
    x = rng.multivariate_normal(loc, S_theta, size=n_obs)
    if flatten:
        return x.flatten()
    return x


def configurator(forward_dict, mode="posterior", scale_data=30.0, as_summary_condition=False):
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
