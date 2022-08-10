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

# Corresponds to Task T.1 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np

bayesflow_benchmark_info = {
    'simulator_is_batched': True,
    'parameter_names': None,
    'configurator_info': 'posterior'
}


def prior(D=10, scale=0.1):
    """ Generates a draw from a D-dimensional Gaussian prior distribution with a 
    spherical convariance matrix given by sigma * I_D. Represents the location vector of
    a (conjugate) Gaussian likelihood.
    
    Parameters
    ----------
    D     : int, optional, default : 10
        The dimensionality of the Gaussian prior distribution.
    scale : float, optional, default : 0.1
        The scale of the Gaussian prior.
        
    Returns
    -------
    theta : np.ndarray of shape (D, )
        A single draw from the D-dimensional Gaussian prior.
    """
    
    return scale*np.random.default_rng().normal(size=D)


def simulator(theta, n_obs=None, scale=0.1):
    """ Generates batched draws from a D-dimenional Gaussian distributions given a batch of 
    location (mean) parameters of D dimensions. Assumes a spherical convariance matrix given 
    by scale * I_D. 
    
    Parameters
    ----------
    theta  : np.ndarray of shape (theta, D)
        The location parameters of the Gaussian likelihood.
    n_obs  : int or None, optional, default: None
        The number of observations to draw from the likelihood given the location
        parameter `theta`. If None, a single draw is produced.
    scale : float, optional, default : 0.1
        The scale of the Gaussian likelihood.
    
    Returns
    -------
    x : np.ndarray of shape (theta.shape[0], theta.shape[1]) if n_obs is None,
        else np.ndarray of shape (theta.shape[0], n_obs, theta.shape[1])
        A single draw or a sample from a batch of Gaussians.
    """
    
    if n_obs is None:
        return scale*np.random.default_rng().normal(loc=theta)
    x = scale*np.random.default_rng().normal(loc=theta, size=(n_obs, theta.shape[0], theta.shape[1]))
    return np.transpose(x, (1, 0, 2))

def configurator(forward_dict):
    """ Default configurator to transform outputs of the generative model to BayesFlow-friendly format."""
    pass
