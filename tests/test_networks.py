import unittest

import numpy as np

from bayesflow.networks import ConditionalCouplingLayer, DenseCouplingNet


class TestDenseCouplingBlock(unittest.TestCase):
    """ Ensures the integrity of the posterior network in terms of
    input - output dimensions.
    """

    @classmethod
    def setUpClass(cls):  
        cls.x_dim = 8
        cls.param_dim = 6
        cls.n_obs1 = 100
        cls.n_obs2 = 200
        cls.batch_size = 8
        cls.params = np.random.randn(cls.batch_size, cls.param_dim)
        cls.x1 = np.random.randn(cls.batch_size, cls.n_obs1, cls.x_dim)
        cls.x1 = np.random.randn(cls.batch_size, cls.n_obs2, cls.x_dim)

    def test_params_2d_data2d(self):
        pass

    def test_params_3d_data_2d(self):
        pass

    def test_params_2d_data_3d(self):
        pass

    def test_params_3d_data_3d(self):
        pass

class TestConditionalCouplingLayer(unittest.TestCase):
    """ Ensures the integrity of the posterior network in terms of
    input - output dimensions.
    """

    @classmethod
    def setUpClass(cls):  
        cls.x_dim = 8
        cls.param_dim = 6
        cls.n_obs1 = 100
        cls.n_obs2 = 200
        cls.batch_size = 8
        cls.params = np.random.randn(cls.batch_size, cls.param_dim)
        cls.x1 = np.random.randn(cls.batch_size, cls.n_obs1, cls.x_dim)
        cls.x1 = np.random.randn(cls.batch_size, cls.n_obs2, cls.x_dim)

    def test_forward_inverse_1_layer(self):
        pass

    def test_forward_inverse_6_layers(self):
        pass

    def test_with_default_settings(self):
        pass

    def test_wo_permutation(self):
        pass

    def test_wo_actnorm(self):
        pass