import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats

from bayesflow.exceptions import ConfigurationError


class GaussianMeanSimulator:
    """ Simulates batches of D-variate Gaussians.

    Attributes
    ----------
    D: int
        Dimensionality of D-variate Gaussian
    sigma: np.ndarray
        Covariance matrix
    """

    def __init__(self, D, s=None):
        """

        Parameters
        ----------
        D: int
            Dimensionality of the Gaussian. Must fit the according prior!
        s: int or float or list or np.ndarray, default: None
            Shape of the Gaussian, see examples for different formats

        Examples
        --------

        Initializing a :class:`GaussianSimulator` with default unit scale.

        >>> simulator = GaussianMeanSimulator(D=3)
        >>> simulator.sigma   # np.diag([1, 1, 1])


        Initializing a :class:`GaussianSimulator` with isotropic scale.

        >>> simulator = GaussianMeanSimulator(D=3, s=2.5)
        >>> simulator.sigma   # np.diag([2.5, 2.5, 2.5])


        Initializing a :class:`GaussianSimulator` with custom diagonal scale as `list`.

        >>> simulator = GaussianMeanSimulator(D=3, s=[1, 2, 3])
        >>> simulator.sigma   # np.diag([1.0, 2.0, 3.0])


        Initializing a :class:`GaussianSimulator` with custom diagnonal scale as `np.ndarray`.

        >>> simulator = GaussianMeanSimulator(D=3, s=np.array([1, 2, 3]))
        >>> simulator.sigma   # np.diag([1.0, 2.0, 3.0])

        Initializing a :class:`GaussianSimulator` with a full scale matrix.

        >>> A = np.random.random((3, 3))
        >>> simulator = GaussianMeanSimulator(D=3, s = np.dot(A, A.T))  # AA^T is always positive semi-definite
        >>> simulator.sigma   # full matrix AA^T


        """
        self.D = D

        # Default: Unit variance
        if s is None:
            self.sigma = np.eye(self.D)

        # Unit variance with factor s
        elif isinstance(s, (int, float)):
            self.sigma = np.eye(self.D) * s

        # s is list or np.array: Either custom diagonal or full matrix
        elif isinstance(s, (list, np.ndarray)):
            # cast any list-like input to float np.array
            if isinstance(s, list):
                s = np.array(s)
            s = s.astype(float)

            # Diagonal covariance matrix with different diagonal entries (from s)
            if s.ndim == 1:
                assert len(s) == D, "Must provide D entries in diagonal s!"
                self.sigma = np.diag(s)

            # Full covariance matrix
            elif s.ndim == 2:
                assert s.shape[0] == D and s.shape[1] == D, "Must provide DxD matrix!"
                try:
                    _ = np.linalg.cholesky(s)
                except Exception:
                    raise ConfigurationError("Covariance Matrix must be positive semi-definite!")

                self.sigma = s

    def __call__(self, theta, n_obs):
        """ Generates batches of samples from the D-variate Gaussian

        Parameters
        ----------
        theta: np.ndarray
            Means of the Gaussian

        n_obs: int
            Number of observations per dataset

        Returns
        -------
        sim_data: np.ndarray
            Simulated data, shape (n_sim, n_obs, D)

        """
        n_sim, D = theta.shape
        assert D == self.D

        tril = tf.linalg.cholesky(np.stack([self.sigma] * n_sim))  # tf requires Cholesky decomposed sigma

        mvn = tfp.distributions.MultivariateNormalTriL(loc=theta, scale_tril=tril)

        sim_data = mvn.sample(n_obs)
        sim_data = np.array(sim_data)
        sim_data = np.transpose(sim_data, (1, 0, 2))
        return sim_data


class GaussianMeanCovSimulator:
    def __call__(self, params, n_obs):
        """ Generates batches of samples from the D-variate Gaussian
        Parameters
        ----------
        params : tuple of np.ndarrays
            Means and convariance matrices of the gaussians
        n_obs  : int
            Number of observations per dataset
        Returns
        -------
        sim_data: np.ndarray
            Simulated data, shape (n_sim, n_obs, D)
        """
        means, cov = params
        tril_cov = tf.linalg.cholesky(cov)
        sim_data = tfp.distributions.MultivariateNormalTriL(means, tril_cov).sample(n_obs)
        return tf.transpose(sim_data, (1, 0, 2))


class MultivariateTSimulator:
    def __init__(self, df=10):
        """
        Provides a batch simulator for a multivariate T distribution.

        Parameters
        ----------
        df: int, default: 10
            Degrees of freedom for the multivariate t distribution
        """
        self.df = df

    def simulate_data(self, p_sample, n_obs):
        """
        Returns a single dataset given a sample from the prior.

        Parameters
        ----------
        p_sample: np.ndarray
            One set of parameter samples for one multivariate t distribution
        n_obs: int
            Number of observations
        """

        D = p_sample.shape[0] // 2
        mu, sd = p_sample[:D], p_sample[D:]
        x = stats.multivariate_t(loc=mu, shape=np.diag(sd), df=self.df).rvs(n_obs)
        return x

    def generate_multiple_datasets(self, p_samples, n_obs):
        """ Generates multiple datasets through `BayesianMultivariateT.generate_data()`

        Parameters
        ----------
        p_samples: np.ndarray
            <batch_size> sets of parameter samples, each for one multivariate t distribution.
            Shape (n_sim, D)
        n_obs: int
            Number of observations per set of parameter samples.

        Returns
        -------
        sim_data: np.ndarray
            Simulated batch of data, shape (n_sim, n_obs, D//2)
        """

        batch_size = p_samples.shape[0]
        theta_dim = p_samples.shape[1] // 2
        sim_data = np.zeros((batch_size, n_obs, theta_dim))

        for bi in range(batch_size):
            sim_data[bi] = self.simulate_data(p_samples[bi], n_obs)
        return sim_data.astype(np.float32)

    def __call__(self, p_samples, n_obs=100):
        """ Generates multiple datasets, calls `generate_multiple_datasets()` internally.

        Parameters
        ----------
        p_samples: np.ndarray
            <batch_size> sets of parameter samples, each for one multivariate t distribution.
            Shape (n_sim, D)
        n_obs: int, default: 100
            Number of observations per set of parameter samples.

        Returns
        -------
        sim_data: np.ndarray
            Simulated batch of data, shape (n_sim, n_obs, D//2)
        """

        return self.generate_multiple_datasets(p_samples, n_obs)
