import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
import tensorflow as tf
from functools import partial

import pytest

from bayesflow import computational_utilities
from bayesflow import sensitivity
from bayesflow import simulation

from tests.test_trainers import _create_training_setup, _prior, _simulator

tf.random.set_seed(1)


def _mms_generator_prior(p1, p2):
    prior_ = partial(_prior, D=2, mu=p1, sigma=p2)
    simulator_ = partial(_simulator, scale=1.0)
    generative_model_ = simulation.GenerativeModel(prior_, simulator_, simulator_is_batched=False, skip_test=True)
    return generative_model_


class TestModelMisspecificationSensitivity:
    def setup(self):
        self.trainer = _create_training_setup(mode="posterior")

        # Mock the approximate posterior sampling of the amortizer, return np.ones of shape (n_sim, n_samples)
        self.trainer.amortizer.sample = MagicMock(
            side_effect=lambda input_dict, n_samples: np.ones((input_dict['parameters'].shape[0], n_samples))
        )

        computational_utilities.maximum_mean_discrepancy = Mock(return_value=tf.random.uniform([1]))
        computational_utilities.aggregated_error = Mock(return_value=tf.random.uniform([1]))

        self.p1_config = {"name": r"$\mu_0$ (prior location)",
                          "values": np.linspace(-0.1, 1.1, num=3),
                          "well_specified_value": 0.0
                          }
        self.p2_config = {"name": r"$\tau_0$ (prior scale)",
                          "values": np.linspace(0.1, 4.1, num=4),
                          "well_specified_value": 1.0
                          }

        self.n1, self.n2 = len(self.p1_config['values']), len(self.p2_config['values'])

        self.posterior_error, self.summary_mmd = sensitivity.model_misspecification_sensitivity(
            trainer=self.trainer,
            generator_misspecification=_mms_generator_prior,
            p1_config=self.p1_config,
            p2_config=self.p2_config,
            n_posterior_samples=500,
            n_sim=200
        )

    def test_compute_model_misspecification_sensitivity(self):
        assert self.posterior_error['value'].shape == (self.n1, self.n2)
        assert self.summary_mmd['value'].shape == (self.n1, self.n2)

    def test_plot_grid(self):
        plot_config = {'vmin': 0, 'vmax': 2}
        fig_rmse = sensitivity.plot_grid(data=self.posterior_error,
                                         p1_config=self.p1_config,
                                         p2_config=self.p2_config,
                                         plot_config=plot_config,
                                         type="RMSE")

        # Assert that the array of the color data is equal to the input data array
        figure_pcolor_data = fig_rmse.axes[0].get_children()[0].get_array().reshape(self.n1, self.n2).data
        assert np.array_equal(figure_pcolor_data, self.posterior_error['value'])

        # check for correct labels
        assert fig_rmse.axes[0].get_xlabel() == self.p1_config['name']
        assert fig_rmse.axes[0].get_ylabel() == self.p2_config['name']

        # check for correct cmap limits
        assert fig_rmse.axes[1].get_ylim()[0] == pytest.approx(plot_config['vmin'])
        assert fig_rmse.axes[1].get_ylim()[1] == pytest.approx(plot_config['vmax'])
