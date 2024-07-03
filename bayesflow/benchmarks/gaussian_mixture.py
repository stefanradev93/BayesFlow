import numpy as np


def simulator():
    """Non-configurable simulator running with default settings."""
    prior_draws = prior()
    observables = observation_model(prior_draws)
    return dict(parameters=prior_draws, observables=observables)


def prior(lower_bound: float = -10.0, upper_bound: float = 10.0, D: int = 2, rng: np.random.Generator = None):
    """Generates a random draw from a 2-dimensional uniform prior bounded between
    `lower_bound` and `upper_bound` representing the common mean of a 2D Gaussian
    mixture model (GMM).

    Parameters
    ----------
    lower_bound : float, optional, default : -10
        The lower bound of the uniform prior
    upper_bound : float, optional, default : 10
        The upper bound of the uniform prior
    D           : int, optional, default: 2
        The dimensionality of the mixture model
    rng         : np.random.Generator or None, default: None
        An optional random number generator to use

    Returns
    -------
    params : np.ndarray of shape (D, )
        A single draw from the D-dimensional uniform prior
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low=lower_bound, high=upper_bound, size=D)


def observation_model(
    params: np.ndarray, prob: float = 0.5, scale_c1: float = 1.0, scale_c2: float = 0.1, rng: np.random.Generator = None
):
    """Simulates data from the Gaussian mixture model (GMM) with
    shared location vector. For more details, see

    https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.7

    Important: The parameterization uses scales, so use sqrt(var),
    if you want to be working with variances instead of scales.

    Parameters
    ----------
    params   : np.ndarray of shape (D,)
        The D-dimensional vector of parameter locations.
    prob     : float, optional, default: 0.5
        The mixture probability (coefficient).
    scale_c1 : float, optional, default: 1.
        The scale of the first component
    scale_c2 : float, optional, default: 0.1
        The scale of the second component
    rng      : np.random.Generator or None, default: None
        An optional random number generator to use

    Returns
    -------
    x : np.ndarray of shape (2,)
        The 2D vector generated from the GMM simulator.
    """

    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Draw component index
    idx = rng.binomial(n=1, p=prob)

    # Draw 2D-Gaussian sample according to component index
    if idx == 0:
        return rng.normal(loc=params, scale=scale_c1)
    return rng.normal(loc=params, scale=scale_c2)
