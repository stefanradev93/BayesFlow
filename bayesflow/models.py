import types
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from bayesflow.exceptions import SimulationError, ConfigurationError


class GenerativeModel(object):
    """
    This class is a factory for the different internal implementations of a GenerativeModel:
    - if priors/simulators are passed as a list, we are in the meta setting
            -> initialize MetaGenerativeModel
    - if only one prior/simulator is passed, we are in the parameter estimation setting
            -> initialize SimpleGenerativeModel
    """

    def __new__(cls, *args):
        if any([isinstance(arg, list) for arg in args]):
            g = object.__new__(MetaGenerativeModel)
        else:
            g = object.__new__(SimpleGenerativeModel)
        g.__init__(*args)

        return g

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    @abstractmethod
    def _check_consistency(self):
        raise NotImplementedError


class MetaGenerativeModel(GenerativeModel):
    def __init__(self, model_prior, priors, simulators, param_transform=None, param_padding=None):
        assert len(priors) == len(simulators), "Must provide same number of priors and simulators!"

        self.model_prior = model_prior
        self.generative_models = [GenerativeModel(prior, simulator) for prior, simulator in zip(priors, simulators)]
        self.param_transform = param_transform
        self.n_models = len(self.generative_models)

        self._max_param_length = None
        self._data_dim = None
        self._find_max_param_length_and_data_dim()

        if param_padding is not None:
            self.param_padding = param_padding
        else:
            self.param_padding = lambda x: np.pad(x,
                                                  pad_width=((0, 0), (0, self._max_param_length - x.shape[1])),
                                                  mode='constant')

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

        # Prepare data and params placeholders
        params = np.empty((n_sim, self._max_param_length), dtype=np.float32)
        sim_data = np.empty((n_sim, n_obs, *self._data_dim))

        # Sample model indices
        model_indices = self.model_prior(n_sim, self.n_models)

        # gather model indices and simulate datasets of same model index as batch
        # create frequency table of model indices
        m_idx, n = np.unique(model_indices, return_counts=True)

        # iterate over each unique model index and create all datasets for that model index
        for m_idx, n in zip(m_idx, n):
            # sample batch of same models
            params_, sim_data_ = self.generative_models[m_idx](n, n_obs, **kwargs)

            # sort data back into the batch-sized arrays
            target_indices = np.where(model_indices == m_idx)  # find indices in batch-sized array
            params[target_indices] = self.param_padding(params_)  # apply padding to params if required
            sim_data[target_indices] = sim_data_

        model_indices = tf.keras.utils.to_categorical(model_indices, self.n_models)

        return model_indices.astype(np.float32), params.astype(np.float32), sim_data.astype(np.float32)

    def _find_max_param_length_and_data_dim(self):
        # find max_param_length
        model_indices = list(range(len(self.generative_models)))
        param_lengths = []
        for m_idx in model_indices:
            params_, sim_data_ = self.generative_models[m_idx](1, 200)
            param_lengths.append(params_.shape[1])

            # set data dim once
            if self._data_dim is None:  # assumption: all simulators have same data dim. If not -> max search & 0-padd.
                self._data_dim = sim_data_.shape[2:]  # dim0: n_sim, dim1: n_obs, dim2...x: data_dim

        self._max_param_length = max(param_lengths)

    def _check_consistency(self):
        """
        Performs an internal consistency check with datasets of 10 observations each.
        """
        _n_sim = 16
        _n_obs = 200

        try:
            model_indices, params, sim_data = self(n_sim=_n_sim, n_obs=_n_obs)

            if model_indices.shape[0] != _n_sim:
                raise SimulationError(
                    f"Model indices shape 0 = {model_indices.shape[0]} does not match n_sim = {_n_sim}")
            if params.shape[0] != _n_sim:
                raise SimulationError(f"Parameter shape 0 = {params.shape[0]} does not match n_sim = {_n_sim}")
            if sim_data.shape[0] != _n_sim:
                raise SimulationError(f"sim_data shape 0 = {sim_data.shape[0]} does not match n_sim = {_n_sim}")

        except Exception as err:
            raise SimulationError(repr(err))


class SimpleGenerativeModel(GenerativeModel):
    def __init__(self, prior: callable, simulator: callable):
        """
        Initializes a GenerativeModel instance with a prior and simulator.
        The GenerativeModel class's __call__ method is capable of returning batches even if the prior and/or simulator
        do not work on batches.
        ----------

        Arguments:
        prior : callable -- provides prior parameter values.
                Can either return a single parameter set ("single mode") or a batch of parameter sets ("batch mode")
                !!! IMPORTANT: !!!
                If the prior callable works on batches, it must have the signature prior(n_sim) or prior(batch_size)!

        simulator : callable -- function that takes parameter (single or matrix) and returns dataset(s).
                Can either work on n_sim = 1 or perform batch simulation, n_sim > 1
        """

        if not callable(prior):
            raise ConfigurationError("prior must be callable!")

        # Handle parsing arguments of CPython functions
        if isinstance(prior.__call__, types.MethodWrapperType):
            prior = prior
        else:
            prior = prior.__call__

        if not callable(simulator):
            raise ConfigurationError("simulator must be callable!")

        self.prior = prior
        self.simulator = simulator
        self._set_prior_and_simulator()
        self._check_consistency()

    def __call__(self, n_sim, n_obs, **kwargs):
        """
        Simulates n_sim datasets of n_obs observations from the provided simulator with parameters from the prior
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

    def _set_prior_and_simulator(self):
        """
        Priors and simulators can be provided with or withour batch capabilities.
        This function checks if the
        """
        # maybe just do it with try/except too
        prior_args = self.prior.__code__.co_varnames
        if 'n_sim' in prior_args or 'batch_size' in prior_args:
            self.prior = self.prior  # prior already produces batches.
        else:
            self._single_prior = self.prior
            self.prior = lambda n_sim: np.array([self._single_prior() for _ in range(n_sim)])

        _n_sim = 2
        _n_obs = 200
        _params = self.prior(_n_sim)

        try:
            _sim_data = self.simulator(_params, _n_obs)
            self.simulator = self.simulator  # simulator already produces batches.
        except Exception:
            _sim_data = np.array([self.simulator(_params[i], _n_obs) for i in range(_n_sim)])
            self._single_simulator = self.simulator
            self.simulator = lambda params, n_obs, **kwargs:\
                np.array([self._single_simulator(theta, n_obs, **kwargs) for theta in params])


    def _check_consistency(self):
        """
        Performs an internal consistency check with 2 datasets of 100 observations each.
        """
        _n_sim = 2
        _n_obs = 200
        try:
            params, sim_data = self(n_sim=_n_sim, n_obs=_n_obs)
            if params.shape[0] != _n_sim:
                raise SimulationError(f"Parameter shape 0 = {params.shape[0]} does not match n_sim = {_n_sim}")
            if sim_data.shape[0] != _n_sim:
                raise SimulationError(f"sim_data shape 0 = {sim_data.shape[0]} does not match n_sim = {_n_sim}")

        except Exception as err:
            raise SimulationError(repr(err))
