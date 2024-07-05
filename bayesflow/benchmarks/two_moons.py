import numpy as np
from .benchmark import Benchmark

class TwoMoonsBenchmark(Benchmark):

    # def simulator():
    #     """Non-configurable simulator running with default settings."""
    #     prior_draws = prior()
    #     observables = observation_model(prior_draws)
    #     return dict(parameters=prior_draws, observables=observables)

    def prior(self, lower_bound: float = -1.0, upper_bound: float = 1.0, rng: np.random.Generator = None):
        """Generates a random draw from a 2-dimensional uniform prior bounded between
        `lower_bound` and `upper_bound` which represents the two parameters of the two moons simulator.

        Parameters
        ----------
        lower_bound: float, optional, default : -1.
            The lower bound of the uniform prior.
        upper_bound: float, optional, default : 1.
            The upper bound of the uniform prior.
        rng: np.random.Generator or None, default: None
            An optional random number generator to use.

        Returns
        -------
        params: np.ndarray of shape (2, )
            A single draw from the 2-dimensional uniform prior.
        """

        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(low=lower_bound, high=upper_bound, size=2)


    def observation_model(self, params: np.ndarray, rng: np.random.Generator = None):
        """Implements data generation from the two-moons model with a bimodal posterior.
        See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.8

        Parameters
        ----------
        params: np.ndarray of shape (2, )
            The vector of two model parameters.
        rng: np.random.Generator or None, default: None
            An optional random number generator to use.

        Returns
        -------
        observables: np.ndarray of shape (2, )
            The 2D vector generated from the two moons simulator.
        """

        # Use default RNG, if None specified
        if rng is None:
            rng = np.random.default_rng()

        # Generate noise
        alpha = rng.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
        r = rng.normal(loc=0.1, scale=0.01)

        # Forward process
        rhs1 = np.array([r * np.cos(alpha) + 0.25, r * np.sin(alpha)])
        rhs2 = np.array([-np.abs(params[0] + params[1]) / np.sqrt(2.0), (-params[0] + params[1]) / np.sqrt(2.0)])

        return rhs1 + rhs2
