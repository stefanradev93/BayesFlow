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


def observation_model(params: np.ndarray, T: int = 100, scale_by_T: bool = True, rng: np.random.Generator = None):
    """Simulates data from the custom Bernoulli GLM likelihood, see
    https://arxiv.org/pdf/2101.04653.pdf, Task T.5

    Important: `scale_sum` should be set to False if the simulator is used
    with variable `T` during training, otherwise the information of `T` will
    be lost.

    Parameters
    ----------
    params     : np.ndarray of shape (10,)
        The vector of model parameters (`params[0]` is intercept, `params[i], i > 0` are weights).
    T          : int, optional, default: 100
        The simulated duration of the task (eq. the number of Bernoulli draws).
    scale_by_T : bool, optional, default: True
        A flag indicating whether to scale the summayr statistics by T.
    rng        : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (10,)
        The vector of sufficient summary statistics of the data.
    """

    # Use default RNG, if None provided
    if rng is None:
        rng = np.random.default_rng()

    # Unpack parameters
    beta, f = params[0], params[1:]

    # Generate design matrix
    V = rng.normal(size=(9, T))

    # Draw from Bernoulli GLM
    z = rng.binomial(n=1, p=expit(V.T @ f + beta))

    # Compute and return (scaled) sufficient summary statistics
    x1 = np.sum(z)
    x_rest = V @ z
    x = np.append(x1, x_rest)
    if scale_by_T:
        x /= T
    return x
