import numpy as np

from bayesflow.exceptions import SimulationError


class GenerativeModel:
    def __init__(self, prior, simulator):
        self.prior = prior
        self.simulator = simulator

        self._check_consistency()

    def __call__(self, n_sim, n_obs, **kwargs):
        """
        Simulates n_sim datasets of n_obs observations from the provided simulator
        ----------

        Arguments:
        n_sim : int -- number of simulation to perform at the given step (i.e., batch size)
        n_obs : int or callable -- if int, then treated as a fixed number of observations, if callable, then
                                   treated as a function for sampling N, i.e., N ~ p(N)
        ----------
        Returns:
        params    : np.array (np.float32) of shape (n_sim, param_dim) -- array of sampled parameters
        sim_data  : np.array (np.float32) of shape (n_sim, n_obs, data_dim) -- array of simulated data sets

        """
        params = self.prior(n_sim)
        sim_data = self.simulator(params, n_obs, **kwargs)

        return params.astype(np.float32), sim_data.astype(np.float32)

    def _check_consistency(self, _n_sim=1, _n_obs=10):
        try:
            params, sim_data = self(n_sim=_n_sim, n_obs=_n_obs)
            if params.shape[0] != _n_sim:
                raise SimulationError(f"Parameter shape 0 = {params.shape[0]} does not match n_sim = {_n_sim}")
            if sim_data.shape[0] != _n_sim:
                raise SimulationError(f"sim_data shape 0 = {sim_data.shape[0]} does not match n_sim = {_n_sim}")
            if sim_data.shape[1] != _n_obs:
                raise SimulationError(f"sim_data shape 1 = {sim_data.shape[1]} does not match n_obs = {_n_obs}")

        except Exception as err:
            raise SimulationError(repr(err))
