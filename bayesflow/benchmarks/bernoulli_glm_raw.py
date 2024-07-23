import numpy as np
from scipy.special import expit


# Global covariance matrix computed once for efficiency
F = np.zeros((9, 9))
for i in range(9):
    F[i, i] = 1 + np.sqrt(i / 9)
    if i >= 1:
        F[i, i - 1] = -2
    if i >= 2:
        F[i, i - 2] = 1
Cov = np.linalg.inv(F.T @ F)


def simulator():
    """Non-configurable simulator running with default settings."""
    prior_draws = prior()
    observables = observation_model(prior_draws)
    return dict(parameters=prior_draws, observables=observables)


def prior(rng: np.random.Generator = None):
    """Generates a random draw from the custom prior over the 10
    Bernoulli GLM parameters (1 intercept and 9 weights). Uses a
    global covariance matrix `Cov` for the multivariate Gaussian prior
    over the model weights, which is pre-computed for efficiency.

    Parameters
    ----------
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    params : np.ndarray of shape (10,)
        A single draw from the prior.
    """

    if rng is None:
        rng = np.random.default_rng()
    beta = rng.normal(0, 2)
    f = rng.multivariate_normal(np.zeros(9), Cov)
    return np.append(beta, f)


def observation_model(params: np.ndarray, T: int = 100, rng: np.random.Generator = None):
    """Simulates data from the custom Bernoulli GLM likelihood, see:
    https://arxiv.org/pdf/2101.04653.pdf, Task T.6

    Returns the raw Bernoulli data.

    Parameters
    ----------
    params : np.ndarray of shape (10,)
        The vector of model parameters (`params[0]` is intercept, `params[i], i > 0` are weights)
    T      : int, optional, default: 100
        The simulated duration of the task (eq. the number of Bernoulli draws).
    rng    : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (T, 10)
        The full simulated set of Bernoulli draws and design matrix.
        Should be configured with an additional trailing dimension if the data is (properly) to be treated as a set.
    """

    # Use default RNG, if None provided
    if rng is None:
        rng = np.random.default_rng()

    # Unpack parameters
    beta, f = params[0], params[1:]

    # Generate design matrix
    V = rng.normal(size=(9, T))

    # Draw from Bernoulli GLM and return
    z = rng.binomial(n=1, p=expit(V.T @ f + beta))
    return np.c_[np.expand_dims(z, axis=-1), V.T]
