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

# Corresponds to Task T.9 from the paper https://arxiv.org/pdf/2101.04653.pdf

import numpy as np
from scipy.integrate import odeint


def prior():
    """ Generates a draw from a 2-dimensional (independent) lognormal prior
    which represents the contact and recovery rate parameters of a basic SIR model.
    
    Returns
    -------
    theta : np.ndarray of shape (2,)
        A single draw from the 2-dimensional prior.
    """
    
    theta = np.random.default_rng().lognormal(
        mean=[np.log(0.4), np.log(1/8)], 
        sigma=[0.5, 0.2]
    )    
    return theta


def _deriv(x, t, N, beta, gamma):
    """ Helper function for scipy.integrate.odeint."""

    S, I, R = x
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return dS, dI, dR

def simulator(theta, N=1e6, T=160, I0=1., R0=0., subsample=10, total_count=1000):
    """ Runs a SIR model simulation for T time steps and returns `subsample` evenly spaced
    points from the simulated trajectory, given disease parameters (contact and recovery rate) `theta`.

    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.9. 

    Note, that the simulator will scale the outputs between 0 and 1.
    
    Parameters
    ----------
    theta       : np.ndarray of shape (2,)
        The 2-dimensional vector of disease parameters.
    N           : float, optional, default: 1e6 = 1 000 000
        The size of the simulated population.
    T           : T, optional, default: 160
        The duration (time horizon) of the simulation.
    I0          : float, optional, default: 1.
        The number of initially infected individuals.
    R0          : float, optional, default: 0.
        The number of initially recovered individuals.
    subsample   : int or None, optional, default: 10
        The number of evenly spaced time points to return. If None,
        no subsampling will be performed and all T timepoints will be returned.
    total_count : int, optional, default: 1000
        The N parameter of the binomial noise distribution. Used just
        for scaling the data and magnifying the effect of noise, such that
        max infected = total_count.

    Returns
    -------
    x : np.ndarray of shape (subsample,) or (T,) if subsample=None
        The time series of simulated infected individuals.
    """
    
    # Create vector (list) of initial conditions
    x0 = N-I0-R0, I0, R0
    
    # Unpack parameter vector into scalars
    beta, gamma = theta
    
    # Prepate time vector between 0 and T of length T
    t_vec = np.linspace(0, T, T)

    # Integrate using scipy and retain only infected (2-nd dimension)
    irt = odeint(_deriv, x0, t_vec, args=(N, beta, gamma))[:, 1]
    
    # Subsample evenly the specified number of points, if specified
    if subsample is not None:
        irt = irt[::(T // subsample)]
    
    # Add noise, scale and return
    x = np.random.default_rng().binomial(n=total_count, p=irt/N) / total_count
    return x