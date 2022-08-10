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

# Corresponds to Task T.5 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np
from scipy.special import expit

bayesflow_benchmark_info = {
    'generative_model_info':
        {
            'simulator_is_batched': False,
            'parameter_names': [],
        },
    'configurator_info': {}
}

# Global covariance matrix computed once for efficiency
F = np.zeros((9, 9))
for i in range(9):
    F[i, i] = 1 + np.sqrt(i / 9)
    if i >= 1:
        F[i, i-1] = -2
    if i >= 2:
        F[i, i-2] = 1
Cov = np.linalg.inv(F.T@F)


def prior():
    """ Generates a random draw from the custom prior over the 10
    Bernoulli GLM parameters (1 intercept and 9 weights). Uses a
    global covariance matrix `Cov` for the multivariate Gaussian prior
    over the model weights, which is pre-computed for efficiency.
        
    Returns
    -------
    theta : np.ndarray of shape (10,)
        A single draw from the prior.
    """
    
    beta = np.random.default_rng().normal(0, 2)
    f = np.random.default_rng().multivariate_normal(np.zeros(9), Cov)
    return np.append(beta, f)
    

def simulator(theta, T=100):
    """ Simulates data from the custom Bernoulli GLM likelihood, see
    https://arxiv.org/pdf/2101.04653.pdf, Task T.5

    Important: `scale_sum` should be set to False if the simulator is used
    with variable `T` during training, otherwise the information of `T` will
    be lost.

    Parameters
    ----------
    theta     : np.ndarray of shape (10,)
        The vector of model parameters (`theta[0]` is intercept, `theta[i], i > 0` are weights).
    T         : int, optional, default: 100
        The simulated duration of the task (eq. the number of Bernoulli draws).
        
    Returns
    -------
    x : np.ndarray of shape (10,)
        The vector of sufficient summary statistics of the data.
    """
    
    # Unpack parameters
    beta, f = theta[0], theta[1:]

    # Generate design matrix
    V = np.random.default_rng().normal(size=(9, T))

    # Draw from Bernoulli GLM
    z = np.random.default_rng().binomial(n=1, p=expit(V.T @ f + beta))

    # Compute and return sufficient summary statistics
    x1 = np.sum(z)
    x_rest = V@z
    x = np.append(x1, x_rest)   
    return x
    