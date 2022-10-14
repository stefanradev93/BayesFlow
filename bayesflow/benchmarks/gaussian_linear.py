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

# Corresponds to Task T.1 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np

bayesflow_benchmark_info = {
    'simulator_is_batched': True,
    'parameter_names': None,
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


def configurator(forward_dict, mode='posterior'):
    """ Configures simulator outputs for use in BayesFlow training."""

    if mode == 'posterior':
        input_dict = _config_posterior(forward_dict)

    elif mode == 'likelihood':
        input_dict = _config_likelihood(forward_dict)

    elif mode == 'joint':
        input_dict = {}
        input_dict['posterior_inputs'] = _config_posterior(forward_dict)
        input_dict['likelihood_inputs'] = _config_likelihood(forward_dict)
    else:
        raise NotImplementedError('For now, only a choice between ["posterior", "likelihood", "joint"] is available!')
    return input_dict


def _config_posterior(forward_dict):
    """ Helepr function for posterior configuration."""

    input_dict = {}
    input_dict['parameters'] = forward_dict['prior_draws'].astype(np.float32)
    input_dict['direct_conditions'] = forward_dict['sim_data'].astype(np.float32)
    return input_dict


def _config_likelihood(forward_dict):
    """ Helepr function for likelihood configuration."""

    input_dict = {}
    input_dict['conditions'] = forward_dict['prior_draws'].astype(np.float32)
    input_dict['observables'] = forward_dict['sim_data'].astype(np.float32)
    return input_dict
