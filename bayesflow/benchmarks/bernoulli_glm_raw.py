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

# Corresponds to Task T.6 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np
from scipy.special import expit

bayesflow_benchmark_info = {
    "simulator_is_batched": False,
    "parameter_names": [r"$\beta$"] + [r"$f_{}$".format(i) for i in range(1, 10)],
    "configurator_info": "posterior",
}

# Global covariance matrix computed once for efficiency
F = np.zeros((9, 9))
for i in range(9):
    F[i, i] = 1 + np.sqrt(i / 9)
    if i >= 1:
        F[i, i - 1] = -2
    if i >= 2:
        F[i, i - 2] = 1
Cov = np.linalg.inv(F.T @ F)


def prior(rng=None):
    """Generates a random draw from the custom prior over the 10
    Bernoulli GLM parameters (1 intercept and 9 weights). Uses a
    global covariance matrix `Cov` for the multivariate Gaussian prior
    over the model weights, which is pre-computed for efficiency.

    Parameters
    ----------
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    theta : np.ndarray of shape (10,)
        A single draw from the prior.
    """

    if rng is None:
        rng = np.random.default_rng()
    beta = rng.normal(0, 2)
    f = rng.multivariate_normal(np.zeros(9), Cov)
    return np.append(beta, f)


def simulator(theta, T=100, rng=None):
    """Simulates data from the custom Bernoulli GLM likelihood, see:
    https://arxiv.org/pdf/2101.04653.pdf, Task T.6

    Returns the raw Bernoulli data.

    Parameters
    ----------
    theta : np.ndarray of shape (10,)
        The vector of model parameters (`theta[0]` is intercept, `theta[i], i > 0` are weights)
    T     : int, optional, default: 100
        The simulated duration of the task (eq. the number of Bernoulli draws).
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (T, 10)
        The full simulated set of Bernoulli draws and design matrix.
        Should be configured with an additional trailing dimension if the data is (properly) to be treated as a set.
    """

    # Use default RNG, if None provided
    if rng is None:
        rng = np.random.default_rng()

    # Unpack parameters
    beta, f = theta[0], theta[1:]

    # Generate design matrix
    V = rng.normal(size=(9, T))

    # Draw from Bernoulli GLM and return
    z = rng.binomial(n=1, p=expit(V.T @ f + beta))
    return np.c_[np.expand_dims(z, axis=-1), V.T]


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
    # Return 3D output
    if as_summary_condition:
        input_dict["summary_conditions"] = forward_dict["sim_data"].astype(np.float32)
    # Flatten along 2nd and 3rd axis
    else:
        x = forward_dict["sim_data"]
        x = x.reshape(x.shape[0], -1)
        input_dict["direct_conditions"] = x.astype(np.float32)
    return input_dict


def _config_likelihood(forward_dict):
    """Helper function for likelihood configuration."""

    input_dict = {}

    # Create observables (adding a dummy var)
    obs = forward_dict["sim_data"][:, :, 0]
    obs_dummy = np.random.randn(obs.shape[0], obs.shape[1])
    input_dict["observables"] = np.stack([obs, obs_dummy], axis=2).astype(np.float32)

    # Create condition (repeating param draws)
    design = forward_dict["sim_data"][:, :, 1:]
    T = design.shape[1]
    params_rep = np.stack([forward_dict["prior_draws"]] * T, axis=1)
    input_dict["conditions"] = np.concatenate([design, params_rep], axis=-1).astype(np.float32)
    return input_dict
