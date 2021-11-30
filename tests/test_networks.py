import copy
import unittest

import numpy as np
import tensorflow as tf

import tests.example_objects as ex
from bayesflow.models import GenerativeModel
from bayesflow.networks import InvariantNetwork, InvertibleNetwork


class TestInvariantNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generative_model = GenerativeModel(ex.priors.dm_prior, ex.simulators.dm_batch_simulator)
        cls.network = InvariantNetwork()
        params, sim_data = cls.generative_model(64, 128)
        cls.network(sim_data)  # one prediction to initialize layers
        cls.optimizer = tf.keras.optimizers.Adam()

    def test_invariant_network_step(self):
        params, sim_data = self.generative_model(64, 128)
        trainable_variables_before = copy.deepcopy(self.network.trainable_variables)

        with tf.GradientTape() as tape:
            out = self.network(sim_data)
            loss = tf.keras.losses.mean_squared_error(tf.random.normal(out.shape), out)  # invoke arbitrary loss

        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        trainable_variables_after = copy.deepcopy(self.network.trainable_variables)

        # assert that any weights are updated in each layer
        for before, after in zip(trainable_variables_before, trainable_variables_after):
            self.assertTrue(np.any(before != after))


class TestInvertibleNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generative_model = GenerativeModel(ex.priors.dm_prior, ex.simulators.dm_batch_simulator)
        cls.invariant_network = InvariantNetwork()
        cls.network = InvertibleNetwork({'n_params': 5})
        cls.optimizer = tf.keras.optimizers.Adam()

    def test_invertible_network_step(self):
        params, sim_data = self.generative_model(64, 128)

        sim_data = self.invariant_network(sim_data)
        _, _ = self.network(params, sim_data)  # initialize layers
        trainable_variables_before = copy.deepcopy(self.network.trainable_variables)
        with tf.GradientTape() as tape:
            z, log_det_J = self.network(params, sim_data)
            loss = tf.reduce_mean(0.5 * tf.square(tf.norm(z, axis=-1)) - log_det_J) 

        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        trainable_variables_after = copy.deepcopy(self.network.trainable_variables)

        # assert that any weights are updated in each layer
        for before, after in zip(trainable_variables_before, trainable_variables_after):
            self.assertTrue(np.any(before != after))

