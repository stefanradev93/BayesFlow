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

# Corresponds to Task T.5 from the paper https://arxiv.org/pdf/2101.04653.pdf

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


def simulator(theta, T=100, scale_by_T=True, rng=None):
    """Simulates data from the custom Bernoulli GLM likelihood, see
    https://arxiv.org/pdf/2101.04653.pdf, Task T.5

    Important: `scale_sum` should be set to False if the simulator is used
    with variable `T` during training, otherwise the information of `T` will
    be lost.

    Parameters
    ----------
    theta      : np.ndarray of shape (10,)
        The vector of model parameters (`theta[0]` is intercept, `theta[i], i > 0` are weights).
    T          : int, optional, default: 100
        The simulated duration of the task (eq. the number of Bernoulli draws).
    scale_by_T : bool, optional, default: True
        A flag indicating whether to scale the summayr statistics by T.
    rng        : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (10,)
        The vector of sufficient summary statistics of the data.
    """

    # Use default RNG, if None provided
    if rng is None:
        rng = np.random.default_rng()

    # Unpack parameters
    beta, f = theta[0], theta[1:]

    # Generate design matrix
    V = rng.normal(size=(9, T))

    # Draw from Bernoulli GLM
    z = rng.binomial(n=1, p=expit(V.T @ f + beta))

    # Compute and return (scaled) sufficient summary statistics
    x1 = np.sum(z)
    x_rest = V @ z
    x = np.append(x1, x_rest)
    if scale_by_T:
        x /= T
    return x


def configurator(forward_dict, mode="posterior"):
    """Configures simulator outputs for use in BayesFlow training."""

    # Case only posterior configuration
    if mode == "posterior":
        input_dict = _config_posterior(forward_dict)

    # Case only likelihood configuration
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
