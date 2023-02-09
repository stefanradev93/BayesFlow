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

# Corresponds to Task T.8 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np

bayesflow_benchmark_info = {
    "simulator_is_batched": False,
    "parameter_names": [r"$\theta_1$", r"$\theta_2$"],
    "configurator_info": "posterior",
}


def prior(lower_bound=-1.0, upper_bound=1.0, rng=None):
    """Generates a random draw from a 2-dimensional uniform prior bounded between
    `lower_bound` and `upper_bound` which represents the two parameters of the two moons simulator.

    Parameters
    ----------
    lower_bound : float, optional, default : -1
        The lower bound of the uniform prior.
    upper_bound : float, optional, default : 1
        The upper bound of the uniform prior.
    rng         : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    theta : np.ndarray of shape (2,)
        A single draw from the 2-dimensional uniform prior.
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low=lower_bound, high=upper_bound, size=2)


def simulator(theta, rng=None):
    """Implements data generation from the two-moons model with a bimodal posterior.
    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.8

    Parameters
    ----------
    theta   : np.ndarray of shape (2,)
        The vector of two model parameters.
    rng     : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (2,)
        The 2D vector generated from the two moons simulator.
    """

    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Generate noise
    alpha = rng.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
    r = rng.normal(loc=0.1, scale=0.01)

    # Forward process
    rhs1 = np.array([r * np.cos(alpha) + 0.25, r * np.sin(alpha)])
    rhs2 = np.array([-np.abs(theta[0] + theta[1]) / np.sqrt(2.0), (-theta[0] + theta[1]) / np.sqrt(2.0)])

    return rhs1 + rhs2


def configurator(forward_dict, mode="posterior"):
    """Configures simulator outputs for use in BayesFlow training."""

    # Case only posterior configuration
    if mode == "posterior":
        input_dict = _config_posterior(forward_dict)

    # Case only plikelihood configuration
    elif mode == "likelihood":
        input_dict = _config_likelihood(forward_dict)

    # Case posterior and likelihood configuration (i.e., joint inference)
    elif mode == "joint":
        input_dict = {}
        input_dict["posterior_inputs"] = _config_posterior(forward_dict)
        input_dict["likelihood_inputs"] = _config_likelihood(forward_dict)

    # Throw otherwise
    else:
        raise NotImplementedError('For now, only a choice between ["posterior", "likelihood", "joint"] is available!')
    return input_dict


def _config_posterior(forward_dict):
    """Helper function for posterior configuration."""

    input_dict = {}
    input_dict["parameters"] = forward_dict["prior_draws"].astype(np.float32)
    input_dict["direct_conditions"] = forward_dict["sim_data"].astype(np.float32)
    return input_dict


def _config_likelihood(forward_dict):
    """Helper function for likelihood configuration."""

    input_dict = {}
    input_dict["conditions"] = forward_dict["prior_draws"].astype(np.float32)
    input_dict["observables"] = forward_dict["sim_data"].astype(np.float32)
    return input_dict
