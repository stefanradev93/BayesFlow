from unittest import TestCase
import numpy as np

from bayesflow.models import GenerativeModel, SimpleGenerativeModel, MetaGenerativeModel
import tests.example_objects_for_tests as ex


class TestGenerativeModel(TestCase):
    def test_simple_generative_model(self):
        generative_model = GenerativeModel(ex.dm_prior, ex.dm_batch_simulator)
        _, _ = generative_model(n_sim=10, n_obs=10)

    def test_meta_generative_model(self):
        M = 10
        D = 10
        prior = ex.TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [ex.MultivariateT(df) for df in np.arange(1, 101, M)]
        generative_model = GenerativeModel(ex.model_prior, priors, simulators)
        _, _, _ = generative_model(n_sim=10, n_obs=10)

    def test_meta_generative_model_different_param_shapes(self):
        priors = [ex.model1_params_prior, ex.model2_params_prior, ex.model3_params_prior]
        simulators = [ex.forward_model1, ex.forward_model2, ex.forward_model3]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)
        _, _, _ = generative_model(n_sim=10, n_obs=10)


class TestSimpleGenerativeModel(TestCase):
    def test_init(self):
        generative_model = SimpleGenerativeModel(ex.dm_prior, ex.dm_batch_simulator)
        return generative_model

    def test_simulation(self):
        _n_sim = 10
        _n_obs = 10
        generative_model = self.test_init()
        params, data = generative_model(n_sim=_n_sim, n_obs=_n_obs)
        # todo validate shapes


class TestMetaGenerativeModel(TestCase):
    def test_init(self):
        M = 10
        D = 10
        prior = ex.TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [ex.MultivariateT(df) for df in np.arange(1, 101, M)]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)

        return generative_model

    def test_init_different_param_shapes(self):
        priors = [ex.model1_params_prior, ex.model2_params_prior, ex.model3_params_prior]
        simulators = [ex.forward_model1, ex.forward_model2, ex.forward_model3]
        generative_model = MetaGenerativeModel(ex.model_prior, priors, simulators)

        return generative_model

    def test_simulation(self):
        _n_sim = 10
        _n_obs = 100
        generative_model = self.test_init()
        model_idx, params, data = generative_model(n_sim=_n_sim, n_obs=_n_obs)
        # todo validate shapes

    def test_simulation_different_param_shapes(self):
        _n_sim = 10
        _n_obs = 100
        generative_model = self.test_init_different_param_shapes()
        model_idx, params, data = generative_model(n_sim=_n_sim, n_obs=_n_obs)
        # todo validate shapes
