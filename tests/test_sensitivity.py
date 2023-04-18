from functools import partial
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import tensorflow as tf

from copy import deepcopy

from bayesflow import computational_utilities, sensitivity, simulation
from tests.test_trainers import _create_training_setup, _prior, _simulator

tf.random.set_seed(1)


def _mms_generator_prior(p1, p2):
    prior_ = partial(_prior, D=2, mu=p1, sigma=p2)
    simulator_ = partial(_simulator, scale=1.0)
    generative_model_ = simulation.GenerativeModel(prior_, simulator_, simulator_is_batched=False, skip_test=True)
    return generative_model_


def _trainer_amortizer_sample_mock(input_dict, n_samples):
    n_data_sets, n_params = input_dict["parameters"].shape
    return np.ones(shape=(n_data_sets, n_samples, n_params))


class TestMisspecificationExperiment:
    def setup(self):
        self.trainer = _create_training_setup(mode="posterior")

        # Mock the approximate posterior sampling of the amortizer, return np.ones of shape (n_sim, n_samples)
        self.trainer.amortizer.sample = MagicMock(side_effect=_trainer_amortizer_sample_mock)

        computational_utilities.maximum_mean_discrepancy = Mock(return_value=tf.random.uniform([1]))
        computational_utilities.aggregated_error = Mock(return_value=tf.random.uniform([1]))

        self.p1_config = {
            "name": r"$\mu_0$ (prior location)",
            "values": np.linspace(-0.1, 1.1, num=3),
            "well_specified_value": 0.0,
        }
        self.p2_config = {
            "name": r"$\tau_0$ (prior scale)",
            "values": np.linspace(0.1, 4.1, num=4),
            "well_specified_value": 1.0,
        }

        self.n1, self.n2 = len(self.p1_config["values"]), len(self.p2_config["values"])

        self.posterior_error, self.summary_mmd = sensitivity.misspecification_experiment(
            trainer=self.trainer,
            generator=_mms_generator_prior,
            first_config_dict=self.p1_config,
            second_config_dict=self.p2_config,
            n_posterior_samples=50,
            n_sim=5,
        )

    def test_compute_model_misspecification_sensitivity(self):
        assert self.posterior_error["values"].shape == (self.n1, self.n2)
        assert self.summary_mmd["values"].shape == (self.n1, self.n2)


    def test_plot_model_misspecification_sensitivity(self):
        plot_config = {"vmin": 0, "vmax": 2}
        fig_rmse = sensitivity.plot_model_misspecification_sensitivity(
            results_dict=self.posterior_error,
            first_config_dict=self.p1_config,
            second_config_dict=self.p2_config,
            plot_config=plot_config
        )

        # Assert that the array of the color data is equal to the input data array
        figure_pcolor_data = fig_rmse.axes[0].get_children()[0].get_array().reshape(self.n1, self.n2).data
        assert np.array_equal(figure_pcolor_data, self.posterior_error["values"])

        # check for correct labels
        assert fig_rmse.axes[0].get_xlabel() == self.p1_config["name"]
        assert fig_rmse.axes[0].get_ylabel() == self.p2_config["name"]

        # check for correct cmap limits
        assert fig_rmse.axes[1].get_ylim()[0] == pytest.approx(plot_config["vmin"])
        assert fig_rmse.axes[1].get_ylim()[1] == pytest.approx(plot_config["vmax"])

    def test_plot_model_misspecification_sensitivity_invalid_name(self):
        results_dict_invalid = deepcopy(self.posterior_error)
        results_dict_invalid['name'] = "invalid_name"
        with pytest.raises(NotImplementedError):
            _ = sensitivity.plot_model_misspecification_sensitivity(
                results_dict=results_dict_invalid,
                first_config_dict=self.p1_config,
                second_config_dict=self.p2_config
            )

    def test_plot_color_grid(self):
        n1 = 2
        n2 = 3
        x_array = np.linspace(0, 1, num=n1)
        y_array = np.linspace(-1, 1, num=n2)
        Y, X = np.meshgrid(y_array, x_array)
        Z = X*Y
        cmap = "inferno"
        vmin = -1
        vmax = 3
        xlabel = "foo"
        ylabel = "bar"
        cbar_title = "baz"

        fig = sensitivity.plot_color_grid(x_grid=X, y_grid=Y, z_grid=Z,
                                          cmap=cmap, vmin=vmin, vmax=vmax,
                                          xlabel=xlabel, ylabel=ylabel, cbar_title=cbar_title)

        # Assert that the array of the color data is equal to the input data array
        figure_pcolor_data = fig.axes[0].get_children()[0].get_array().reshape(n1, n2).data
        assert np.array_equal(figure_pcolor_data, Z)

        # check for correct labels
        assert fig.axes[0].get_xlabel() == xlabel
        assert fig.axes[0].get_ylabel() == ylabel

        # check for correct cmap limits
        assert fig.axes[1].get_ylim()[0] == pytest.approx(vmin)
        assert fig.axes[1].get_ylim()[1] == pytest.approx(vmax)
