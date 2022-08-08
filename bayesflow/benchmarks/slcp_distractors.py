# Copyright 2022 The BayesFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Corresponds to Task T.4 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np
from scipy.stats import multivariate_t


def get_random_student_t(Sigma, dim=2, mu_scale=15.):
    """ A helper function to create a "frozen" multivariate student-t distribution of dimensions `dim`.

    Parameters
    ----------
    Sigma    : np.ndarray of shape (dim, dim)
        The (symmetric) positive semidefinite shape matrix of the student-t distribution.
    dim      : int, optional, default: 2
        The dimensionality of the student-t distribution.
    mu_scale : float, optional, default: 15
        The scale of the zero-centered Gaussian prior from which the mean vector 
        of the student-t distribution is drawn. 
  
    Returns
    -------
    student : callable (scipy.stats._multivariate.multivariate_t_frozen)
        The student-t generator.
    """
    
    # Draw mean
    mu = mu_scale * np.random.default_rng().normal(size=dim)
    
    # Return student-t object
    return multivariate_t(loc=mu, shape=Sigma, df=2)


def draw_mixture_student_t(num_students, Sigma, n_draws=46, dim=2, mu_scale=15.):
    """ Helper function to generate `n_draws` random draws from a mixture of `num_students` 
    multivariate Student-t distributions. 
    
    Uses the function `get_random_student_t` to create each of the studen-t callable objects.

    Parameters
    ----------
    num_students : int
        The number of multivariate student-t mixture components
    Sigma        : np.ndarray of shape (dim, dim)
        The (symmetric) positive semidefinite shared shape matrix of the student-t distributions.
    n_draws      : int, optional, default: 46 
        The number of draws to obtain from the mixture distribution.
    dim          : int, optional, default: 2
        The dimensionality of each student-t distribution in the mixture.
    mu_scale     : float, optional, default: 15
        The scale of the zero-centered Gaussian prior from which the mean vector 
        of each student-t distribution in the mixture is drawn. 
    
    Returns
    -------
    distractors : np.ndarray of shape (n_draws, dim)
        The random draws from the mixture.
    """

    # Obtain a list of scipy frozen distributions (each will have a different mean)
    students = [get_random_student_t(Sigma, dim, mu_scale) for _ in range(num_students)]

    # Obtain the sample of n_draws from the mixture and return 
    sample = [students[np.random.default_rng().integers(low=0, high=num_students)].rvs() for _ in range(n_draws)]

    return np.array(sample)


def prior(lower_bound=-3., upper_bound=3.):
    """ Generates a draw from a 5-dimensional uniform prior bounded between 
    `lower_bound` and `upper_bound`.
    
    Parameters
    ----------
    lower_bound : float, optional, default : -3.
        The lower bound of the uniform prior.
    upper_bound : float, optional, default : 3.
        The upper bound of the uniform prior.
        
    Returns
    -------
    theta : np.ndarray of shape (5, )
        A single draw from the 5-dimensional uniform prior.
    """
    
    return np.random.default_rng().uniform(low=lower_bound, high=upper_bound, size=5)


def simulator(theta, n_obs=4, n_dist=46, dim=2, mu_scale=15., flatten=True):
    """ Implements data generation from the SLCP model with distractors designed as a benchmark
    for a simple likelihood and a complex posterior due to non-linear pushforward theta -> x.
    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.4
    
    Parameters
    ----------
    theta   : np.ndarray of shape (theta, D)
        The location parameters of the Gaussian likelihood.
    n_obs   : int, optional, default: 4
        The number of observations to generate from the slcp likelihood.
    n_dist  : int, optional, default: 46
        The number of distractor to draw from the distractor likelihood.
    dim          : int, optional, default: 2
        The dimensionality of each student-t distribution in the mixture.
    mu_scale     : float, optional, default: 15
        The scale of the zero-centered Gaussian prior from which the mean vector 
        of each student-t distribution in the mixture is drawn. 
    flatten : bool, optional, default: True
        A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
        representation of the simulated data is returned.
    
    Returns
    -------
    x : np.ndarray of shape (n_obs * 2,  + )
    boolean flag.
        The sample of simulated data from the slcp model. 
    """
    
    # Specify 2D location
    loc = np.array([theta[0], theta[1]])
    
    # Specify 2D covariance matrix
    s1 = theta[2] ** 2
    s2 = theta[3] ** 2
    rho = np.tanh(theta[4])
    cov = rho*s1*s2
    S_theta = np.array([[s1**2, cov], [cov, s2**2]])
    
    # Obtain informative part of the data
    x_info = np.random.default_rng().multivariate_normal(loc, S_theta, size=n_obs)

    # Obtain uninformative part of the data
    x_uninfo = draw_mixture_student_t(
        num_students=20, Sigma=S_theta, n_draws=n_dist, dim=dim, mu_scale=mu_scale)

    # Concatenate informative with uninformative and return
    x = np.concatenate([x_info, x_uninfo], axis=0)
    if flatten:
        return x.flatten()
    return x