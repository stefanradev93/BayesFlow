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

# Corresponds to Task T.8 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np

bayesflow_benchmark_info = {
    'simulator_is_batched': False,
    'parameter_names': [r'$\theta_1$', r'$\theta_2$'],
    'configurator_info': 'posterior'
}


def prior(lower_bound=-1., upper_bound=1.):
    """ Generates a draw from a 2-dimensional uniform prior bounded between 
    `lower_bound` and `upper_bound` which represents the two parameters of the two moons simulator
    
    Parameters
    ----------
    lower_bound : float, optional, default : -1.
        The lower bound of the uniform prior.
    upper_bound : float, optional, default : 1.
        The upper bound of the uniform prior.
        
    Returns
    -------
    theta : np.ndarray of shape (2,)
        A single draw from the 2-dimensional uniform prior.
    """
    
    return np.random.default_rng().uniform(low=lower_bound, high=upper_bound, size=2)


def simulator(theta):
    """ Implements data generation from the two-moons model with a bimodal posterior.
    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.8
    
    Parameters
    ----------
    theta   : np.ndarray of shape (2,)
        The vector of two model parameters.
    
    Returns
    -------
    x : np.ndarray of shape (2,)
        The 2D vector generated from the two moons simulator.
    """
    
    # Generate noise
    alpha = np.random.default_rng().uniform(low=-0.5*np.pi, high=0.5*np.pi)
    r = np.random.default_rng().normal(loc=0.1, scale=0.01)
    
    # Forward process
    rhs1 = np.array([
        r*np.cos(alpha) + 0.25, 
        r*np.sin(alpha)
    ])
    rhs2 = np.array([
        -np.abs(theta[0] + theta[1]) / np.sqrt(2.),
        (-theta[0] + theta[1]) / np.sqrt(2.)
    ])
    
    return rhs1 + rhs2


def configurator(forward_dict, mode='posterior'):
    """ Configures simulator outputs for use in BayesFlow training."""

    if mode == 'posterior':
        input_dict = {}
        input_dict['parameters'] = forward_dict['prior_draws'].astype(np.float32)
        input_dict['direct_conditions'] = forward_dict['sim_data'].astype(np.float32)
        return input_dict
    else:
        raise NotImplementedError('For now, only posterior mode is available!')
