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

    # @unittest.skip("Meta-Model Optimization only works with simulator output shape (n_sim, n_obs[, *data_dims])")
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
        _n_sim = 10
        _n_obs = 100
        generative_model = self.test_init()
        params, data = generative_model(n_sim=N_SIM, n_obs=N_OBS)
        # todo validate shapes


class TestMetaGenerativeModel(unittest.TestCase):
    def test_init(self):
        M = 10
        D = 8
        prior = ex.TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [ex.MultivariateT(df) for df in np.arange(1, 101, M)]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)

        return generative_model

    def test_simulation(self):
        generative_model = self.test_init()
        model_idx, params, data = generative_model(n_sim=N_SIM, n_obs=N_OBS)
        # todo validate shapes

    def test_init_different_param_shapes(self):
        priors = [ex.model1_params_prior, ex.model2_params_prior, ex.model3_params_prior]
        simulators = [ex.forward_model1, ex.forward_model2, ex.forward_model3]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)

        return generative_model

    def test_simulation_different_param_shapes(self):
        generative_model = self.test_init_different_param_shapes()
        model_idx, params, data = generative_model(n_sim=16, n_obs=150)
        # todo validate shapes
