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

# Corresponds to Task T.7 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np

bayesflow_benchmark_info = {
    'simulator_is_batched': False,
    'parameter_names': [r'$\mu_1$', r'$\mu_2$'],
    'configurator_info': 'posterior'
}


def prior(lower_bound=-10., upper_bound=10.):
    """ Generates a draw from a 2-dimensional uniform prior bounded between 
    `lower_bound` and `upper_bound` representing the common mean of a 2D Gaussian
    mixture model (GMM).
    
    Parameters
    ----------
    lower_bound : float, optional, default : -10
        The lower bound of the uniform prior.
    upper_bound : float, optional, default : 10
        The upper bound of the uniform prior.
        
    Returns
    -------
    theta : np.ndarray of shape (D, )
        A single draw from the D-dimensional uniform prior.
    """
    
    return np.random.default_rng().uniform(low=lower_bound, high=upper_bound)


def simulator(theta, prob=0.5, scale_c1=1., scale_c2=0.01):
    """ Implements data generation from the Gaussian mixture model (GMM) with
    shared location vector. For more details, see
    
    https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.7
    
    Parameters
    ----------
    theta    : np.ndarray of shape (2,)
        The 2-dimensional vector of parameter locations.
    prob     : float, optional, default: 0.5
        The mixture probability (coefficient).
    scale_c1 : float, optional, default: 1.
        The scale of the first component.
    scale_c2 : float, optional, default: 0.01
        The scale of the second component.
    
    Returns
    -------
    x : np.ndarray of shape (2,)
        The 2D vector generated from the GMM simulator.
    """
    
    # Draw component index
    idx = np.random.default_rng().binomial(n=1, p=prob)
    
    # Draw 2D-Gaussian sample according to component index
    if idx == 0:
        return scale_c1*np.random.default_rng().normal(loc=theta)
    return scale_c2*np.random.default_rng().normal(loc=theta)


def configurator(forward_dict, mode='posterior', scale_data=12):
    """ Configures simulator outputs for use in BayesFlow training."""

    if mode == 'posterior':
        input_dict = {}
        input_dict['parameters'] = forward_dict['prior_draws'].astype(np.float32)
        input_dict['direct_conditions'] = forward_dict['sim_data'].astype(np.float32) / scale_data
        return input_dict
    else:
        raise NotImplementedError('For now, only posterior mode is available!')