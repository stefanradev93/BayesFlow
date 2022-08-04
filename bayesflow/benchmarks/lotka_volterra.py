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

# Corresponds to Task T.10 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np
from scipy.integrate import odeint


def prior():
    """ Generates a draw from a 4-dimensional (independent) lognormal prior
    which represents the four contact parameters of the Lotka-Volterra model.
    
    Returns
    -------
    theta : np.ndarray of shape (4,)
        A single draw from the 4-dimensional prior.
    """
    
    theta = np.random.default_rng().lognormal(
        mean=[-0.125, -3, -0.125, -3], 
        sigma=0.5
    )    
    return theta


def _deriv(x, t, alpha, beta, gamma, delta):
    """ Helper function for scipy.integrate.odeint."""

    X, Y = x
    dX = alpha*X - beta*X*Y
    dY = -gamma*Y + delta*X*Y
    return dX, dY 

def simulator(theta, X0=30, Y0=1, T=20, subsample=10, flatten=True):
    """ Runs a Lotka-Volterra simulation for T time steps and returns `subsample` evenly spaced
    points from the simulated trajectory, given contact parameters `theta`.

    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.10. 
    
    Parameters
    ----------
    theta       : np.ndarray of shape (2,)
        The 2-dimensional vector of disease parameters.
    X0          : float, optional, default: 30
        Initial number of prey species.
    Y0          : float, optional, default: 1
        Initial number of predator species.
    T           : T, optional, default: 20
        The duration (time horizon) of the simulation.
    subsample   : int or None, optional, default: 10
        The number of evenly spaced time points to return. If None,
        no subsampling will be performed and all T timepoints will be returned.
    flatten     : bool, optional, default: True 
        A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
        representation of the simulated data is returned.

    Returns
    -------
    x : np.ndarray of shape (subsample, 2) or (subsample*2,) if `subsample is not None`, 
        otherwise shape (T, 2) or (T*2,) if `subsample is None`.
        The time series of simulated predator and pray populations
    """
    
    # Create vector (list) of initial conditions
    x0 = X0, Y0
    
    # Unpack parameter vector into scalars
    alpha, beta, gamma, delta = theta
    
    # Prepate time vector between 0 and T of length T
    t_vec = np.linspace(0, T, T)
    
    # Integrate using scipy and retain only infected (2-nd dimension)
    pp = odeint(_deriv, x0, t_vec, args=(alpha, beta, gamma, delta))
    
    # Subsample evenly the specified number of points, if specified
    if subsample is not None:
        pp = pp[::(T // subsample)]
    
    # Add noise, decide whether to flatten and return
    x = np.random.default_rng().lognormal(pp, sigma=0.1)
    if flatten:
        return x.flatten()
    return x