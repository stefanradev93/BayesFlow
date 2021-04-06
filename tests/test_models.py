import unittest

import numpy as np

import tests.example_objects as ex
from bayesflow.models import GenerativeModel, SimpleGenerativeModel, MetaGenerativeModel

N_SIM = 16
N_OBS = 20


class TestGenerativeModel(unittest.TestCase):
    def test_simple_generative_model(self):
        generative_model = GenerativeModel(ex.priors.dm_prior, ex.simulators.dm_batch_simulator)
        _params, _sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)

    def test_meta_generative_model(self):
        M = 10
        D = 100
        prior = ex.priors.TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [ex.simulators.MultivariateT(df) for df in np.arange(1, 101, M)]
        generative_model = GenerativeModel(ex.priors.model_prior, priors, simulators)
        _model_indices, _params, _sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)

    def test_meta_generative_model_different_param_shapes(self):
        priors = [ex.priors.model1_params_prior, ex.priors.model2_params_prior, ex.priors.model3_params_prior]
        simulators = [ex.simulators.forward_model1, ex.simulators.forward_model2, ex.simulators.forward_model3]
        generative_model = MetaGenerativeModel(ex.priors.model_prior, priors, simulators)
        _model_indices, _params, _sim_data = generative_model(n_sim=16, n_obs=150)


class TestSimpleGenerativeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generative_model = SimpleGenerativeModel(ex.priors.dm_prior, ex.simulators.dm_batch_simulator)
        cls.generative_model = generative_model

    def test_simulation(self):
        generative_model = self.generative_model
        params, sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)
        self.assertTrue(params.shape[0] == N_SIM)
        self.assertTrue(sim_data.shape[0] == N_SIM)
        self.assertTrue(sim_data.shape[1] == N_OBS)


class TestMetaGenerativeModel(unittest.TestCase):
    @classmethod
    def init_same_param_shapes(cls):
        M = 10
        D = 8
        prior = ex.priors.TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [ex.simulators.MultivariateT(df) for df in np.arange(1, 101, M)]
        generative_model = MetaGenerativeModel(ex.priors.model_prior, priors, simulators)
        return generative_model

    @classmethod
    def init_different_param_shapes(cls):
        priors = [ex.priors.model1_params_prior, ex.priors.model2_params_prior, ex.priors.model3_params_prior]
        simulators = [ex.simulators.forward_model1, ex.simulators.forward_model2, ex.simulators.forward_model3]
        generative_model = MetaGenerativeModel(ex.priors.model_prior, priors, simulators)
        return generative_model

    @classmethod
    def setUpClass(cls):
        cls.generative_model_same_param_shapes = cls.init_same_param_shapes()
        cls.generative_model_different_param_shapes = cls.init_different_param_shapes()

    def test_simulation_same_param_shapes(self):
        generative_model = self.generative_model_same_param_shapes

        model_idx, params, sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)
        self.assertTrue(model_idx.shape[0] == N_SIM)
        self.assertTrue(params.shape[0] == N_SIM)
        self.assertTrue(sim_data.shape[0] == N_SIM)
        self.assertTrue(sim_data.shape[1] == N_OBS)

    def test_simulation_different_param_shapes(self):
        _n_sim = 16
        _n_obs = 150
        generative_model = self.generative_model_different_param_shapes

        model_indices, params, sim_data = generative_model(n_sim=_n_sim, n_obs=_n_obs)
        self.assertTrue(model_indices.shape[0] == _n_sim)
        self.assertTrue(params.shape[0] == _n_sim)
        self.assertTrue(sim_data.shape[0] == _n_sim)
        self.assertTrue(sim_data.shape[1] == _n_obs)
