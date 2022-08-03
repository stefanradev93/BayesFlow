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

def simulator(theta):
    """ Implements data generation from the SLCP model with distractors designed as a benchmark
    for a simple likelihood and a complex posterior due to non-linear pushforward theta -> x.
    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.4
    
     Parameters
    ----------
    theta   : np.ndarray of shape (theta, D)
        The location parameters of the Gaussian likelihood.
    n_obs   : int
        The number of observations to generate from the slcp likelihood.
    
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
    S = np.array([[s1**2, cov], [cov, s2**2]])
    
    # Obtain informative part
    x_info = np.random.default_rng().multivariate_normal(loc, S, size=4).flatten()

    # Obtain uninformative part
   # TODO
    return x_info