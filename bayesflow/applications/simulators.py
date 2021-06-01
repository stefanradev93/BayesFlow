import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesflow.exceptions import ConfigurationError


class GaussianSimulator:
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

        >>> simulator = GaussianSimulator(D=3)
        >>> simulator.sigma   # np.diag([1, 1, 1])


        Initializing a :class:`GaussianSimulator` with isotropic scale.

        >>> simulator = GaussianSimulator(D=3, s=2.5)
        >>> simulator.sigma   # np.diag([2.5, 2.5, 2.5])


        Initializing a :class:`GaussianSimulator` with custom diagonal scale as `list`.

        >>> simulator = GaussianSimulator(D=3, s=[1, 2, 3])
        >>> simulator.sigma   # np.diag([1.0, 2.0, 3.0])


        Initializing a :class:`GaussianSimulator` with custom diagnonal scale as `np.ndarray`.

        >>> simulator = GaussianSimulator(D=3, s=np.array([1, 2, 3]))
        >>> simulator.sigma   # np.diag([1.0, 2.0, 3.0])

        Initializing a :class:`GaussianSimulator` with a full scale matrix.

        >>> A = np.random.random((3, 3))
        >>> simulator = GaussianSimulator(D=3, s = np.dot(A, A.T))  # AA^T is always positive semi-definite
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
