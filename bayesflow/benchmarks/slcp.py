import numpy as np


def simulator():
    """Non-configurable simulator running with default settings."""
    prior_draws = prior()
    observables = observation_model(prior_draws)
    return dict(parameters=prior_draws, observables=observables)


def prior(lower_bound: float = -3.0, upper_bound: float = 3.0, rng: np.random.Generator = None):
    """Generates a random draw from a 5-dimensional uniform prior bounded between
    `lower_bound` and `upper_bound` which represents the 5 parameters of the SLCP
    simulator.

    Parameters
    ----------
    lower_bound : float, optional, default : -3
        The lower bound of the uniform prior.
    upper_bound : float, optional, default : 3
        The upper bound of the uniform prior.
    rng         : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    params : np.ndarray of shape (5, )
        A single draw from the 5-dimensional uniform prior.
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low=lower_bound, high=upper_bound, size=5)


def observation_model(params: np.ndarray, n_obs: int = 4, flatten: bool = True, rng: np.random.Generator = None):
    """Generates data from the SLCP model designed as a benchmark for a simple likelihood
    and a complex posterior due to a non-linear pushforward params -> x.

    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.3

    Parameters
    ----------
    params  : np.ndarray of shape (params, D)
        The location parameters of the Gaussian likelihood.
    n_obs   : int, optional, default: 4
        The number of observations to generate from the slcp likelihood.
    flatten : bool, optional, default: True
        A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
        representation of the simulated data is returned.
    rng     : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (n_obs*2, ) or (n_obs, 2), as indictated by the `flatten`
        boolean flag. The sample of simulated data from the SLCP model.
    """

    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Specify 2D location
    loc = np.array([params[0], params[1]])

    # Specify 2D covariance matrix
    s1 = params[2] ** 2
    s2 = params[3] ** 2
    rho = np.tanh(params[4])
    cov = rho * s1 * s2
    S_param = np.array([[s1**2, cov], [cov, s2**2]])

    # Obtain given number of draws from the MVN likelihood
    x = rng.multivariate_normal(loc, S_param, size=n_obs)
    if flatten:
        return x.flatten()
    return x
