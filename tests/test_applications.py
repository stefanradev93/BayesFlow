import unittest

import numpy as np

from bayesflow.applications.priors import GaussianMeanPrior, GaussianMeanCovPrior
from bayesflow.applications.simulators import GaussianMeanSimulator, GaussianMeanCovSimulator


class TestGaussianMeanPrior(unittest.TestCase):
    def test_gaussian_mean_prior(self):
        n_sim = 10
        D = 3
        prior = GaussianMeanPrior(D=D)
        theta = prior(n_sim)
        self.assertTrue((theta.shape == n_sim, D))


class TestGaussianMeanSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_obs = 8
        cls.D = 3
        cls.n_sim = 10
        prior = GaussianMeanPrior(cls.D)
        cls.theta = prior(cls.n_sim)

    def test_unit_sigma(self):
        simulator = GaussianMeanSimulator(self.D)
        true_sigma = np.eye(self.D)
        self.assertTrue(np.array_equal(true_sigma, simulator.sigma))
        sim_data = simulator(self.theta, self.n_obs)
        self.assertTrue((sim_data.shape == self.n_sim, self.n_obs, self.D))

    def test_isotropic_sigma(self):
        s = 2.5
        simulator = GaussianMeanSimulator(self.D, s=s)
        true_sigma = np.eye(self.D) * s
        self.assertTrue(np.array_equal(true_sigma, simulator.sigma))
        sim_data = simulator(self.theta, self.n_obs)
        self.assertTrue((sim_data.shape == self.n_sim, self.n_obs, self.D))

    def test_diagonal_sigma_list_int(self):
        s = [1, 2, 3]
        simulator = GaussianMeanSimulator(self.D, s=s)
        true_sigma = np.array([[1, 0, 0],
                               [0, 2, 0],
                               [0, 0, 3]], dtype=float)
        self.assertTrue(np.array_equal(true_sigma, simulator.sigma))
        sim_data = simulator(self.theta, self.n_obs)
        self.assertTrue((sim_data.shape == self.n_sim, self.n_obs, self.D))

    def test_diagonal_sigma_list_float(self):
        s = [1.0, 2.0, 3.0]
        simulator = GaussianMeanSimulator(self.D, s=s)
        true_sigma = np.array([[1, 0, 0],
                               [0, 2, 0],
                               [0, 0, 3]], dtype=float)
        self.assertTrue(np.array_equal(true_sigma, simulator.sigma))
        sim_data = simulator(self.theta, self.n_obs)
        self.assertTrue((sim_data.shape == self.n_sim, self.n_obs, self.D))

    def test_diagonal_sigma_numpy_int(self):
        s = np.array([1, 2, 3])
        simulator = GaussianMeanSimulator(self.D, s=s)
        true_sigma = np.array([[1, 0, 0],
                               [0, 2, 0],
                               [0, 0, 3]], dtype=float)
        self.assertTrue(np.array_equal(true_sigma, simulator.sigma))
        sim_data = simulator(self.theta, self.n_obs)
        self.assertTrue((sim_data.shape == self.n_sim, self.n_obs, self.D))

    def test_diagonal_sigma_numpy_float(self):
        s = np.array([1.0, 2.0, 3.0])
        simulator = GaussianMeanSimulator(self.D, s=s)
        true_sigma = np.array([[1, 0, 0],
                               [0, 2, 0],
                               [0, 0, 3]], dtype=float)
        self.assertTrue(np.array_equal(true_sigma, simulator.sigma))
        sim_data = simulator(self.theta, self.n_obs)
        self.assertTrue((sim_data.shape == self.n_sim, self.n_obs, self.D))

    def test_full_sigma_numpy(self):
        s = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        simulator = GaussianMeanSimulator(self.D, s=s)
        true_sigma = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]], dtype=float)
        self.assertTrue(np.array_equal(true_sigma, simulator.sigma))
        sim_data = simulator(self.theta, self.n_obs)
        self.assertTrue((sim_data.shape == self.n_sim, self.n_obs, self.D))


class TestGaussianMeanCovPrior(unittest.TestCase):
    def test_gaussian_mean_cov_prior(self):
        n_sim = 10
        D = 3
        prior = GaussianMeanCovPrior(D=D, a0=10, b0=1, m0=0, beta0=1)
        means, cov = prior(n_sim)
        self.assertTrue((means.shape == n_sim, D))
        self.assertTrue((cov.shape == n_sim, D, D))


class TestGaussianMeanCovSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_obs = 8
        cls.D = 3
        cls.n_sim = 10
        prior = GaussianMeanCovPrior(cls.D, a0=10, b0=1, m0=0, beta0=1)
        cls.theta = prior(cls.n_sim)    # cls.theta = means, cov

    def test_gaussian_mean_cov_simulator(self):
        simulator = GaussianMeanCovSimulator()
        sim_data = simulator(self.theta, self.n_obs)
        self.assertTrue((sim_data.shape == self.n_sim, self.n_obs, self.D))