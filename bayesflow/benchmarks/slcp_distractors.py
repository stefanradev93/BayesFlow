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

# Corresponds to Task T.4 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np
from scipy.stats import multivariate_t

bayesflow_benchmark_info = {
    "simulator_is_batched": False,
    "parameter_names": [r"$\theta_{}$".format(i) for i in range(1, 6)],
    "configurator_info": "posterior",
}


def get_random_student_t(dim=2, mu_scale=15, shape_scale=0.01, rng=None):
    """A helper function to create a "frozen" multivariate student-t distribution of dimensions `dim`.

    Parameters
    ----------
    dim          : int, optional, default: 2
        The dimensionality of the student-t distribution.
    mu_scale     : float, optional, default: 15
        The scale of the zero-centered Gaussian prior from which the mean vector
        of the student-t distribution is drawn.
    shape_scale  : float, optional, default: 0.01
        The scale of the assumed `np.eye(dim)` shape matrix. The default is chosen to keep
        the scale of the distractors and observations relatively similar.
    rng          : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    student : callable (scipy.stats._multivariate.multivariate_t_frozen)
        The student-t generator.
    """

    # Use default RNG, if None provided
    if rng is None:
        rng = np.random.default_rng()

    # Draw mean
    mu = mu_scale * rng.normal(size=dim)

    # Return student-t object
    return multivariate_t(loc=mu, shape=shape_scale, df=2, allow_singular=True, seed=rng)


def draw_mixture_student_t(num_students, n_draws=46, dim=2, mu_scale=15.0, rng=None):
    """Helper function to generate `n_draws` random draws from a mixture of `num_students`
    multivariate Student-t distributions.

    Uses the function `get_random_student_t` to create each of the studen-t callable objects.

    Parameters
    ----------
    num_students : int
        The number of multivariate student-t mixture components
    n_draws      : int, optional, default: 46
        The number of draws to obtain from the mixture distribution.
    dim          : int, optional, default: 2
        The dimensionality of each student-t distribution in the mixture.
    mu_scale     : float, optional, default: 15
        The scale of the zero-centered Gaussian prior from which the mean vector
        of each student-t distribution in the mixture is drawn.
    rng          : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    sample : np.ndarray of shape (n_draws, dim)
        The random draws from the mixture of students.
    """

    # Use default RNG, if None provided
    if rng is None:
        rng = np.random.default_rng()

    # Obtain a list of scipy frozen distributions (each will have a different mean)
    students = [get_random_student_t(dim, mu_scale, rng=rng) for _ in range(num_students)]

    # Obtain the sample of n_draws from the mixture and return
    sample = [students[rng.integers(low=0, high=num_students)].rvs() for _ in range(n_draws)]

    return np.array(sample)


def prior(lower_bound=-3.0, upper_bound=3.0, rng=None):
    """Generates a random draw from a 5-dimensional uniform prior bounded between
    `lower_bound` and `upper_bound`.

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


def simulator(theta, n_obs=4, n_dist=46, dim=2, mu_scale=15.0, flatten=True, rng=None):
    """Generates data from the SLCP model designed as a benchmark for a simple likelihood
    and a complex posterior due to a non-linear pushforward theta -> x. In addition, it
    outputs uninformative distractor data.

    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.4

    Parameters
    ----------
    theta    : np.ndarray of shape (theta, D)
        The location parameters of the Gaussian likelihood.
    n_obs    : int, optional, default: 4
        The number of observations to generate from the slcp likelihood.
    n_dist   : int, optional, default: 46
        The number of distractor to draw from the distractor likelihood.
    dim      : int, optional, default: 2
        The dimensionality of each student-t distribution in the mixture.
    mu_scale : float, optional, default: 15
        The scale of the zero-centered Gaussian prior from which the mean vector
        of each student-t distribution in the mixture is drawn.
    flatten  : bool, optional, default: True
        A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
        representation of the simulated data is returned.
    rng      : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (n_obs*2 + n_dist*2,) if `flatten=True`, otherwise
        np.ndarray of shape (n_obs + n_dist, 2) if `flatten=False`
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

    # Obtain informative part of the data
    x_info = rng.multivariate_normal(loc, S_theta, size=n_obs)

    # Obtain uninformative part of the data
    x_uninfo = draw_mixture_student_t(num_students=20, n_draws=n_dist, dim=dim, mu_scale=mu_scale, rng=rng)

    # Concatenate informative with uninformative and return
    x = np.concatenate([x_info, x_uninfo], axis=0)
    if flatten:
        return x.flatten()
    return x


def configurator(forward_dict, mode="posterior", scale_data=50.0, as_summary_condition=False):
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
