import unittest

import numpy as np

import tests.example_objects as ex
from bayesflow.applications.priors import model_prior, TPrior, GaussianMeanCovPrior
from bayesflow.applications.simulators import MultivariateTSimulator, GaussianMeanCovSimulator
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
        prior = TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [MultivariateTSimulator(df) for df in np.arange(1, 101, M)]
        generative_model = GenerativeModel(model_prior, priors, simulators)
        _model_indices, _params, _sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)

    def test_meta_generative_model_different_param_shapes(self):
        priors = [ex.priors.model1_params_prior, ex.priors.model2_params_prior, ex.priors.model3_params_prior]
        simulators = [ex.simulators.forward_model1, ex.simulators.forward_model2, ex.simulators.forward_model3]
        generative_model = MetaGenerativeModel(model_prior, priors, simulators)
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

    def test_param_transform(self):
        def param_transform(x):
            return np.exp(x)

        generative_model = SimpleGenerativeModel(ex.priors.dm_prior, ex.simulators.dm_batch_simulator,
                                                 param_transform=param_transform)
        _params, _sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)

    def test_data_transform(self):
        def data_transform(x):
            noise = 0.001 * np.random.random(x.shape)
            return x + noise

        generative_model = SimpleGenerativeModel(prior=ex.priors.dm_prior, simulator=ex.simulators.dm_batch_simulator,
                                                 data_transform=data_transform)
        _params, _sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)

    def test_param_and_data_transform(self):
        def param_transform(x):
            return np.exp(x)

        def data_transform(x):
            noise = 0.001 * np.random.random(x.shape)
            return x + noise

        generative_model = SimpleGenerativeModel(prior=ex.priors.dm_prior, simulator=ex.simulators.dm_batch_simulator,
                                                 param_transform=param_transform, data_transform=data_transform)
        _params, _sim_data = generative_model(n_sim=N_SIM, n_obs=N_OBS)


class TestMetaGenerativeModel(unittest.TestCase):
    @classmethod
    def init_same_param_shapes(cls):
        M = 10
        D = 8
        prior = TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [MultivariateTSimulator(df) for df in np.arange(1, 101, M)]
        generative_model = MetaGenerativeModel(model_prior, priors, simulators)
        return generative_model

    @classmethod
    def init_different_param_shapes(cls):
        priors = [ex.priors.model1_params_prior, ex.priors.model2_params_prior, ex.priors.model3_params_prior]
        simulators = [ex.simulators.forward_model1, ex.simulators.forward_model2, ex.simulators.forward_model3]
        generative_model = MetaGenerativeModel(model_prior, priors, simulators)
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

    def test_same_param_and_data_transform(self):
        def data_transform(x):
            noise = 0.001 * np.random.random(x.shape)
            return x + noise

        def param_transform(x):
            return np.exp(x)

        M = 10
        D = 8
        prior = TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [MultivariateTSimulator(df) for df in np.arange(1, 101, M)]

        _generative_model = MetaGenerativeModel(model_prior=model_prior, priors=priors, simulators=simulators,
                                                param_transforms=param_transform, data_transforms=data_transform)

    def test_same_param_transform(self):
        def param_transform(x):
            return np.exp(x)

        M = 10
        D = 8
        prior = TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [MultivariateTSimulator(df) for df in np.arange(1, 101, M)]

        _generative_model = MetaGenerativeModel(model_prior=model_prior, priors=priors, simulators=simulators,
                                                param_transforms=param_transform)

    def test_same_data_transform(self):
        def data_transform(x):
            noise = 0.001 * np.random.random(x.shape)
            return x + noise

        M = 10
        D = 8
        prior = TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [MultivariateTSimulator(df) for df in np.arange(1, 101, M)]

        _generative_model = MetaGenerativeModel(model_prior=model_prior, priors=priors, simulators=simulators,
                                                data_transforms=data_transform)

    def test_individual_param_and_data_transform(self):
        param_transforms = [lambda x: np.exp(x),
                            None,
                            lambda x: np.round(x, 3)]

        data_transforms = [lambda x: x + np.random.random(x.shape),
                           lambda x: np.exp(x),
                           None]

        M = 3
        D = 4
        prior = TPrior(D // 2, mu_scale=1.0, scale_scale=5.0)
        priors = [prior] * M
        simulators = [MultivariateTSimulator(df) for df in np.round(np.linspace(1, 101, M))]

        _generative_model = MetaGenerativeModel(model_prior=model_prior, priors=priors, simulators=simulators,
                                                param_transforms=param_transforms, data_transforms=data_transforms)

    def test_structural_param_transform_tuple_to_numpy(self):
        D = 5

        def param_transform_mvn(theta):
            means, cov = theta
            var = np.diagonal(cov, axis1=1, axis2=2)
            return np.concatenate([means, var], axis=1)

        prior = GaussianMeanCovPrior(D=D, a0=10, b0=1, m0=0, beta0=1)
        simulator = GaussianMeanCovSimulator()
        generative_model = GenerativeModel(prior, simulator, param_transform=param_transform_mvn)
