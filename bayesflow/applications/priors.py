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

import numpy as np
from scipy import stats


class GaussianMeanPrior:
    """ Provides a gaussian prior for means of a D-variate Gaussian.
    Attributes
    ----------
    D        : int
        Dimensionality of multivariate Gaussian
    mu_mean  : float, default: 0.0
        Mean of mu prior
    mu_scale : float, default: 1.0
        Scale of mu prior
    """

    def __init__(self, D, mu_mean=0.0, mu_scale=1.0):
        self.D = D
        self.mu_mean = mu_mean
        self.mu_scale = mu_scale

    def __call__(self):
        """ Generates n_sim sets of parameter draws.
     
        Returns
        -------
        theta : np.ndarray
            Sampled parameters, shape (D, )
        """
        theta = np.random.default_rng().normal(self.mu_mean, self.mu_scale, size=self.D)
        return theta

    def log_density(self, prior_draws):
        """ Computes prior density of the Gaussian prior. 

        Parameters
        -------

        prior_draws : np.ndarray
            The prior draws for which to compute the log pdf, shape (batch_size, D)

        Returns
        -------
        lpdf : np.ndarray
            The log pdf of the prior draws, shape (batch_size, )
        
        """

        return stats.norm(loc=self.mu_mean, scale=self.mu_scale).logpdf(prior_draws)


class StudentTPrior:
    def __init__(self, D, mu_scale, scale_scale):
        """ Provides a prior for the parameters of a multivariate t distribution.
        Parameters
        ----------
        D           : int
            Dimensionality of the multivariate t
        mu_scale    : float
            Scale of the prior over the location parameter.
        scale_scale : float
            Scale of the prior over the scale parameter.
        """

        self.theta_dim = D
        self.prior_mu = stats.multivariate_normal(np.zeros(self.D), mu_scale * np.eye(self.D))
        self.prior_scale = stats.uniform(0, scale_scale)

    def __call__(self):
        """ Generates a random draw from the prior.

        Returns
        -------
        theta: np.array
            Sampled parameters, shape (D, )
        """

        mu_samples = self.prior_mu.rvs()
        scale_samples = self.prior_scale.rvs(self.D)
        theta = np.r_[mu_samples, scale_samples].astype(np.float32)
        return theta