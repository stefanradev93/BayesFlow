import numpy as np
from scipy import stats


def model_prior(batch_size, n_models=3, p_vals=None):
    """
    Samples from the models' prior batch size times and converts to one-hot.
    Assumes equal model priors.

    Parameters
    ----------
    batch_size : int
        The number of samples to draw from the prior
    n_models: int, default: 3
        Number of models
    p_vals: np.ndarray, default: None
        Weights for model prior, defaults to uniform


    Returns
    -------
    m_true : np.ndarray
        the sampled batch of model indices, shape (batch_size, n_models)
    """

    # Equal priors, if nothing specified
    if p_vals is None:
        p_vals = [1 / n_models] * n_models
    m_idx = np.random.choice(n_models, size=batch_size, p=p_vals).astype(np.int32)
    return m_idx


class GaussianPrior:
    """ Provides a gaussian prior for means of a D-variate Gaussian.

    Attributes
    ----------
    D : int
        Dimensionality of multivariate Gaussian
    mu_mean : float, default: 0.0
        Mean of mu prior
    mu_scale : float, default: 1.0
        Scale of mu prior
    """

    def __init__(self, D, mu_mean=0.0, mu_scale=1.0):
        self.D = D
        self.mu_mean = mu_mean
        self.mu_scale = mu_scale

    def __call__(self, n_sim):
        """ Generates n_sim sets of parameter draws.

        Parameters
        ----------
        n_sim: int
            Batch size

        Returns
        -------
        theta : np.ndarray
            Sampled parameters, shape (n_sim, D)

        """
        theta = np.random.default_rng().normal(self.mu_mean, self.mu_scale, size=(n_sim, self.D))
        return theta


class TPrior:
    def __init__(self, theta_dim, mu_scale, scale_scale):
        """ Provides a prior for the parameters of a multivariate t distribution.

        Parameters
        ----------
        theta_dim: int
            Dimensionality of the multivariate t
        mu_scale: float
            Scale of the prior over the location parameter.
        scale_scale: float
            Scale of the prior over the scale parameter.
        """

        self.theta_dim = theta_dim
        self.prior_mu = stats.multivariate_normal(np.zeros(self.theta_dim), mu_scale * np.eye(self.theta_dim))
        self.prior_scale = stats.uniform(0, scale_scale)

    def __call__(self, batch_size):
        """Returns a sample from the prior.

        Parameters
        ----------
        batch_size: int
            Batch size for the parameter batch

        Returns
        -------
        theta: np.array
            Sampled parameters, shape (n_sim, D)

        """

        mu_samples = self.prior_mu.rvs(batch_size)
        if batch_size == 1:
            mu_samples = mu_samples[np.newaxis]
        scale_samples = self.prior_scale.rvs((batch_size, self.theta_dim))
        theta = np.c_[mu_samples, scale_samples].astype(np.float32)
        return theta
