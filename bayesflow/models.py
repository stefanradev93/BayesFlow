from abc import abstractmethod

import numpy as np
import tensorflow as tf

from bayesflow.exceptions import SimulationError, ConfigurationError

import types


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
    def __init__(self, model_prior, priors, simulators, param_transform=None):
        assert len(priors) == len(simulators), "Must provide same number of priors and simulators!"

        self.model_prior = model_prior
        self.generative_models = [GenerativeModel(prior, simulator) for prior, simulator in zip(priors, simulators)]
        self.param_transform = param_transform
        self.n_models = len(self.generative_models)

        self.max_param_length = None

        self._find_max_param_length()
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
        # Sample model indices
        model_indices = self.model_prior(n_sim, self.n_models)

        # Prepare data and params placeholders
        params = []
        sim_data = []

        # todo If the user-provided simulators support batches, group the model_indices array by model indices and
        # sample data for same models as batch. then, just sort the batch elements in the right spots of sim_data
        # Loop for n_sim number of simulations
        for sim_idx in range(n_sim):
            # Simulate from model
            params_, sim_data_ = self.generative_models[model_indices[sim_idx]](1, n_obs, **kwargs)

            # zero padding
            params.append(np.pad(params_[0], pad_width=(0, self.max_param_length - params_.shape[1]), mode='constant'))
            sim_data.append(sim_data_)

        # Convert to numpy arrays
        model_indices = tf.keras.utils.to_categorical(model_indices, self.n_models)

        params = np.array(params)
        sim_data = np.concatenate(sim_data, axis=0)

        return model_indices.astype(np.float32), params.astype(np.float32), sim_data.astype(np.float32)

    def _find_max_param_length(self):
        # find max_param_length
        model_indices = list(range(len(self.generative_models)))
        param_lengths = []
        for m_idx in model_indices:
            params_, _ = self.generative_models[m_idx](1, 1)
            param_lengths.append(params_.shape[1])
        self.max_param_length = max(param_lengths)

    def _check_consistency(self):
        """
        Performs an internal consistency check with datasets of 10 observations each.
        """
        _n_sim = 10
        _n_obs = 10

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
            self.prior = prior
        else:
            self.prior = prior.__call__

        prior_args = self.prior.__code__.co_varnames  # add __call__ because arguments will be checked
        if 'n_sim' in prior_args or 'batch_size' in prior_args:
            self.mode = 'batch'
        else:
            self.mode = 'single'

        if not callable(simulator):
            raise ConfigurationError("simulator must be callable!")
        self.simulator = simulator

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
        if self.mode == 'batch':
            params = self.prior(n_sim)
            sim_data = self.simulator(params, n_obs, **kwargs)

        elif self.mode == 'single':
            params = np.array([self.prior() for i in range(n_sim)])
            try:
                sim_data = self.simulator(params, n_obs, **kwargs)
            except Exception as err:
                sim_data = np.array([self.simulator(params[i], n_obs, **kwargs) for i in range(n_sim)])

        return params.astype(np.float32), sim_data.astype(np.float32)

    def _check_consistency(self):
        """
        Performs an internal consistency check with 2 datasets of 100 observations each.
        """
        _n_sim = 2
        _n_obs = 100
        try:
            params, sim_data = self(n_sim=_n_sim, n_obs=_n_obs)
            if params.shape[0] != _n_sim:
                raise SimulationError(f"Parameter shape 0 = {params.shape[0]} does not match n_sim = {_n_sim}")
            if sim_data.shape[0] != _n_sim:
                raise SimulationError(f"sim_data shape 0 = {sim_data.shape[0]} does not match n_sim = {_n_sim}")

        except Exception as err:
            raise SimulationError(repr(err))
