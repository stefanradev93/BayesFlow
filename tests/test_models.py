import unittest

import numpy as np

import tests.example_objects_for_tests as ex
from bayesflow.models import GenerativeModel, SimpleGenerativeModel, MetaGenerativeModel

N_SIM = 16
N_OBS = 20


class TestGenerativeModel(unittest.TestCase):
    def test_simple_generative_model(self):
        generative_model = GenerativeModel(ex.dm_prior, ex.dm_batch_simulator)
        _, _ = generative_model(n_sim=N_SIM, n_obs=N_OBS)

    def test_meta_generative_model(self):
        M = 10
        D = 100
        prior = ex.TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [ex.MultivariateT(df) for df in np.arange(1, 101, M)]
        generative_model = GenerativeModel(ex.model_prior, priors, simulators)
        _, _, _ = generative_model(n_sim=N_SIM, n_obs=N_OBS)

    def test_meta_generative_model_different_param_shapes(self):
        priors = [ex.model1_params_prior, ex.model2_params_prior, ex.model3_params_prior]
        simulators = [ex.forward_model1, ex.forward_model2, ex.forward_model3]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)
        _, _, _ = generative_model(n_sim=16, n_obs=150)


class TestSimpleGenerativeModel(unittest.TestCase):
    def test_init(self):
        generative_model = SimpleGenerativeModel(ex.dm_prior, ex.dm_batch_simulator)
        return generative_model

    def test_simulation(self):
        generative_model = SimpleGenerativeModel(ex.dm_prior, ex.dm_batch_simulator)
        params, sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)
        self.assertTrue(params.shape[0] == N_SIM)
        self.assertTrue(sim_data.shape[0] == N_SIM)
        self.assertTrue(sim_data.shape[1] == N_OBS)


class TestMetaGenerativeModel(unittest.TestCase):
    def test_init(self):
        M = 10
        D = 8
        prior = ex.TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [ex.MultivariateT(df) for df in np.arange(1, 101, M)]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)

    def test_simulation(self):
        M = 10
        D = 8
        prior = ex.TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [ex.MultivariateT(df) for df in np.arange(1, 101, M)]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)

        model_idx, params, sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)
        self.assertTrue(model_idx.shape[0] == N_SIM)
        self.assertTrue(params.shape[0] == N_SIM)
        self.assertTrue(sim_data.shape[0] == N_SIM)
        self.assertTrue(sim_data.shape[1] == N_OBS)

    def test_init_different_param_shapes(self):
        priors = [ex.model1_params_prior, ex.model2_params_prior, ex.model3_params_prior]
        simulators = [ex.forward_model1, ex.forward_model2, ex.forward_model3]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)

    def test_simulation_different_param_shapes(self):
        _n_sim = 16
        _n_obs = 150
        priors = [ex.model1_params_prior, ex.model2_params_prior, ex.model3_params_prior]
        simulators = [ex.forward_model1, ex.forward_model2, ex.forward_model3]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)

        model_idx, params, sim_data = generative_model(n_sim=_n_sim, n_obs=_n_obs)
        self.assertTrue(model_idx.shape[0] == _n_sim)
        self.assertTrue(params.shape[0] == _n_sim)
        self.assertTrue(sim_data.shape[0] == _n_sim)
        self.assertTrue(sim_data.shape[1] == _n_obs)
