import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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