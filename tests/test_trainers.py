import unittest

import numpy as np

import tests.example_objects as ex
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.models import GenerativeModel
from bayesflow.networks import InvariantNetwork, InvertibleNetwork
from bayesflow.trainers import ParameterEstimationTrainer


class TestParameterEstimationTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        summary_net = InvariantNetwork()
        inference_net = InvertibleNetwork({'n_params': 5})
        amortizer = SingleModelAmortizer(inference_net, summary_net)
        generative_model = GenerativeModel(ex.priors.dm_prior, ex.simulators.dm_batch_simulator)
        trainer = ParameterEstimationTrainer(amortizer, generative_model)
        cls.generative_model = generative_model
        cls.trainer = trainer

    def test_offline_learning(self):
        n_sim = 5000
        n_obs = 100
        true_params, sim_data = self.generative_model(n_sim, n_obs)
        _losses = self.trainer.train_offline(epochs=1, batch_size=64, params=true_params, sim_data=sim_data)

    def test_simulate_and_train_offline(self):
        _losses = self.trainer.simulate_and_train_offline(n_sim=500, epochs=2, batch_size=32, n_obs=100)

    def test_train_online_fixed_n_obs(self):
        _losses = self.trainer.train_online(epochs=2, iterations_per_epoch=100, batch_size=32, n_obs=100)

    def test_train_online_variable_n_obs(self):
        _losses = self.trainer.train_online(epochs=2, iterations_per_epoch=100, batch_size=32,
                                            n_obs=np.random.randint(60, 301))

    def test_train_rounds(self):
        _losses = self.trainer.train_rounds(epochs=1, rounds=5, sim_per_round=200, batch_size=32, n_obs=100)

    def test_train_experience_replay(self):
        _losses = self.trainer.train_experience_replay(epochs=2,
                                                       batch_size=32,
                                                       iterations_per_epoch=100,
                                                       capacity=100,
                                                       n_obs=np.random.randint(60, 301))

