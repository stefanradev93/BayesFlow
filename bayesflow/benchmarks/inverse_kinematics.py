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

# Corresponds to Task 1 from the paper https://arxiv.org/pdf/2101.10763.pdf


import numpy as np

bayesflow_benchmark_info = {"simulator_is_batched": False, "parameter_names": None, "configurator_info": "posterior"}


def prior(scales=None, rng=None):
    """Generates a random draw from a 4-dimensional Gaussian prior distribution with a
    spherical convariance matrix. The parameters represent a robot's arm configuration,
    with the first parameter indicating the arm's height and the remaining three are
    angles.

    Parameters
    ----------
    scales : np.ndarray of shape (4,) or None, optional, default : None
        The four scales of the Gaussian prior.
        If ``None`` provided, the scales from https://arxiv.org/pdf/2101.10763.pdf
        will be used: [0.25, 0.5, 0.5, 0.5]
    rng    : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    theta : np.ndarray of shape (4, )
        A single draw from the 4-dimensional Gaussian prior.
    """

    if rng is None:
        rng = np.random.default_rng()
    if scales is None:
        scales = np.array([0.25, 0.5, 0.5, 0.5])
    return rng.normal(loc=0, scale=scales)


def simulator(theta, l1=0.5, l2=0.5, l3=1.0, **kwargs):
    """Returns the 2D coordinates of a robot arm given parameter vector.
    The first parameter represents the arm's height and the remaining three
    correspond to angles.

    Parameters
    ----------
    theta    : np.ndarray of shape (theta, )
        The four model parameters which will determine the coordinates
    l1       : float, optional, default: 0.5
        The length of the first segment
    l2       : float, optional, default: 0.5
        The length of the second segment
    l3       : float, optional, default: 1.0
        The length of the third segment
    **kwargs : dict, optional, default: {}
        Used for comptability with the other benchmarks, as the model is deterministic

    Returns
    -------
    x : np.ndarray of shape (2, )
        The 2D coordinates of the arm
    """

    # Determine 2D position
    x1 = l1 * np.sin(theta[1])
    x1 += l2 * np.sin(theta[1] + theta[2])
    x1 += l3 * np.sin(theta[1] + theta[2] + theta[3]) + theta[0]
    x2 = l1 * np.cos(theta[1])
    x2 += l2 * np.cos(theta[1] + theta[2])
    x2 += l3 * np.cos(theta[1] + theta[2] + theta[3])
    return np.array([x1, x2])


def configurator(forward_dict, mode="posterior"):
    """Configures simulator outputs for use in BayesFlow training."""

    # Case only posterior configuration
    if mode == "posterior":
        input_dict = _config_posterior(forward_dict)

    # Case only likelihood configuration
    elif mode == "likelihood":
        input_dict = _config_likelihood(forward_dict)

    # Case posterior and likelihood configuration
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
