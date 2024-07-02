import numpy as np
from scipy.integrate import odeint


def simulator():
    """Non-configurable simulator running with default settings."""
    prior_draws = prior()
    observables = observation_model(prior_draws)
    return dict(parameters=prior_draws, observables=observables)


def prior(rng: np.random.Generator = None):
    """Generates a random draw from a 4-dimensional (independent) lognormal prior
    which represents the four contact parameters of the Lotka-Volterra model.

    Parameters
    ----------
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    theta : np.ndarray of shape (4,)
        A single draw from the 4-dimensional prior.
    """

    if rng is None:
        rng = np.random.default_rng()

    theta = rng.lognormal(mean=[-0.125, -3, -0.125, -3], sigma=0.5)
    return theta


def _deriv(x, t, alpha, beta, gamma, delta):
    """Helper function for scipy.integrate.odeint."""

    X, Y = x
    dX = alpha * X - beta * X * Y
    dY = -gamma * Y + delta * X * Y
    return dX, dY


def observation_model(
    theta: np.ndarray,
    X0: int = 30,
    Y0: int = 1,
    T: int = 20,
    subsample: int = 10,
    flatten: bool = True,
    obs_noise: float = 0.1,
    rng: np.random.Generator = None,
):
    """Runs a Lotka-Volterra simulation for T time steps and returns `subsample` evenly spaced
    points from the simulated trajectory, given contact parameters `theta`.

    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.10.

    Parameters
    ----------
    theta       : np.ndarray of shape (2,)
        The 2-dimensional vector of disease parameters.
    X0          : int, optional, default: 30
        Initial number of prey species.
    Y0          : int, optional, default: 1
        Initial number of predator species.
    T           : int, optional, default: 20
        The duration (time horizon) of the simulation.
    subsample   : int or None, optional, default: 10
        The number of evenly spaced time points to return. If None,
        no subsampling will be performed and all T timepoints will be returned.
    flatten     : bool, optional, default: True
        A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
        representation of the simulated data is returned.
    obs_noise   : float, optional, default: 0.1
        The standard deviation of the log-normal likelihood.
    rng         : np.random.Generator or None, default: None
        An optional random number generator to use.

    Returns
    -------
    x : np.ndarray of shape (subsample, 2) or (subsample*2,) if `subsample is not None`,
        otherwise shape (T, 2) or (T*2,) if `subsample is None`.
        The time series of simulated predator and pray populations
    """

    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Create vector (list) of initial conditions
    x0 = X0, Y0

    # Unpack parameter vector into scalars
    alpha, beta, gamma, delta = theta

    # Prepate time vector between 0 and T of length T
    t_vec = np.linspace(0, T, T)

    # Integrate using scipy and retain only infected (2-nd dimension)
    pp = odeint(_deriv, x0, t_vec, args=(alpha, beta, gamma, delta))

    # Subsample evenly the specified number of points, if specified
    if subsample is not None:
        pp = pp[:: (T // subsample)]

    # Ensure minimum count is 0, which will later pass by log(0 + 1)
    pp[pp < 0] = 0.0

    # Add noise, decide whether to flatten and return
    x = rng.lognormal(np.log1p(pp), sigma=obs_noise)
    if flatten:
        return x.flatten()
    return x
