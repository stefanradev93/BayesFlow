# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pytest
from pandas import DataFrame

from bayesflow.amortizers import AmortizedLikelihood, AmortizedPosterior, AmortizedPosteriorLikelihood
from bayesflow.networks import DeepSet, InvertibleNetwork
from bayesflow.simulation import GenerativeModel
from bayesflow.trainers import Trainer


def _prior(D=2, mu=0.0, sigma=1.0):
    """Helper minimal prior function."""
    return np.random.default_rng().normal(loc=mu, scale=sigma, size=D)


def _simulator(theta, n_obs=10, scale=1.0):
    """Helper minimal simulator function."""
    return np.random.default_rng().normal(loc=theta, scale=scale, size=(n_obs, theta.shape[0]))


def _create_training_setup(mode):
    """Helper function to create a relevant training setup."""

    # Create a generative model
    model = GenerativeModel(_prior, _simulator, name="test", simulator_is_batched=False)

    # Case posterior inference
    if mode == "posterior":
        summary_net = DeepSet()
        inference_net = InvertibleNetwork(num_params=2, num_coupling_layers=2)
        amortizer = AmortizedPosterior(inference_net, summary_net)

    # Case likelihood inference
    elif mode == "likelihood":
        surrogate_net = InvertibleNetwork(num_params=2, num_coupling_layers=2)
        amortizer = AmortizedLikelihood(surrogate_net)

    # Case joint inference
    else:
        summary_net = DeepSet()
        inference_net = InvertibleNetwork(num_params=2, num_coupling_layers=2)
        p_amortizer = AmortizedPosterior(inference_net, summary_net)
        surrogate_net = InvertibleNetwork(num_params=2, num_coupling_layers=2)
        l_amortizer = AmortizedLikelihood(surrogate_net)
        amortizer = AmortizedPosteriorLikelihood(p_amortizer, l_amortizer)

    # Create and return trainer instance
    trainer = Trainer(generative_model=model, amortizer=amortizer)
    return trainer

class TestTrainer:
    def setup(self):
        trainer_posterior = _create_training_setup("posterior")
        trainer_likelihood = _create_training_setup("likelihood")
        trainer_joint = _create_training_setup("joint")
        self.trainers = {
            "posterior": trainer_posterior,
            "likelihood": trainer_likelihood,
            "joint": trainer_joint
        }


    @pytest.mark.parametrize("mode", ["posterior", "likelihood"])
    @pytest.mark.parametrize("reuse_optimizer", [True, False])
    @pytest.mark.parametrize("validation_sims", [20, None])
    def test_train_online(self, mode, reuse_optimizer, validation_sims):
        """Tests the online training functionality."""

        # Create trainer and train online
        trainer = self.trainers[mode]
        h = trainer.train_online(
            epochs=2,
            iterations_per_epoch=3,
            batch_size=8,
            use_autograph=False,
            reuse_optimizer=reuse_optimizer,
            validation_sims=validation_sims,
        )

        # Assert (non)-existence of optimizer
        if reuse_optimizer:
            assert trainer.optimizer is not None
        else:
            assert trainer.optimizer is None

        # Ensure losses were stored in the correct format
        if validation_sims is None:
            assert type(h) is DataFrame
        else:
            assert type(h) is dict
            assert type(h["train_losses"]) is DataFrame
            assert type(h["val_losses"]) is DataFrame


    @pytest.mark.parametrize("mode", ["posterior", "joint"])
    @pytest.mark.parametrize("reuse_optimizer", [True, False])
    @pytest.mark.parametrize("validation_sims", [20, None])
    def test_train_experience_replay(self, mode, reuse_optimizer, validation_sims):
        """Tests the experience replay training functionality."""

        # Create trainer and train with experience replay
        trainer = self.trainers[mode]
        h = trainer.train_experience_replay(
            epochs=3, iterations_per_epoch=4, batch_size=8, validation_sims=validation_sims, reuse_optimizer=reuse_optimizer
        )

        # Assert (non)-existence of optimizer
        if reuse_optimizer:
            assert trainer.optimizer is not None
        else:
            assert trainer.optimizer is None

        # Ensure losses were stored in the correct format
        if validation_sims is None:
            assert type(h) is DataFrame
        else:
            assert type(h) is dict
            assert type(h["train_losses"]) is DataFrame
            assert type(h["val_losses"]) is DataFrame


    @pytest.mark.parametrize("mode", ["likelihood", "joint"])
    @pytest.mark.parametrize("reuse_optimizer", [True, False])
    @pytest.mark.parametrize("validation_sims", [20, None])
    def test_train_offline(self, mode, reuse_optimizer, validation_sims):
        """Tests the offline training functionality."""

        # Create trainer and data and train offline
        trainer = self.trainers[mode]
        simulations = trainer.generative_model(100)
        h = trainer.train_offline(
            simulations_dict=simulations,
            epochs=2,
            batch_size=16,
            use_autograph=True,
            validation_sims=validation_sims,
            reuse_optimizer=reuse_optimizer,
        )

        # Assert (non)-existence of optimizer
        if reuse_optimizer:
            assert trainer.optimizer is not None
        else:
            assert trainer.optimizer is None

        # Ensure losses were stored in the correct format
        if validation_sims is None:
            assert type(h) is DataFrame
        else:
            assert type(h) is dict
            assert type(h["train_losses"]) is DataFrame
            assert type(h["val_losses"]) is DataFrame


    @pytest.mark.parametrize("mode", ["likelihood", "posterior"])
    @pytest.mark.parametrize("reuse_optimizer", [True, False])
    @pytest.mark.parametrize("validation_sims", [20, None])
    def test_train_rounds(self, mode, reuse_optimizer, validation_sims):
        """Tests the offline training functionality."""

        # Create trainer and data and train offline
        trainer = self.trainers[mode]
        h = trainer.train_rounds(
            rounds=2,
            sim_per_round=32,
            epochs=2,
            batch_size=8,
            validation_sims=validation_sims,
            reuse_optimizer=reuse_optimizer,
        )

        # Assert (non)-existence of optimizer
        if reuse_optimizer:
            assert trainer.optimizer is not None
        else:
            assert trainer.optimizer is None

        # Ensure losses were stored in the correct format
        if validation_sims is None:
            assert type(h) is DataFrame
        else:
            assert type(h) is dict
            assert type(h["train_losses"]) is DataFrame
            assert type(h["val_losses"]) is DataFrame

    @pytest.mark.parametrize("reference_data", [None, "dict", "numpy"])
    @pytest.mark.parametrize("observed_data_type", ["dict", "numpy"])
    @pytest.mark.parametrize("bootstrap", [True, False])
    def mmd_hypothesis_test_no_reference(self, reference_data, observed_data_type, bootstrap):
        trainer = self.trainers["posterior"]
        _ = trainer.train_online(epochs=1, iterations_per_epoch=1, batch_size=4)

        num_reference_simulations = 10
        num_observed_simulations = 2
        num_null_samples = 5

        if reference_data is None:
            if reference_data == "dict":
                reference_data = trainer.configurator(trainer.generative_model(num_reference_simulations))
            elif reference_data == "numpy":
                reference_data = trainer.configurator(trainer.generative_model(num_reference_simulations))['summary_conditions']

        if observed_data_type == "dict":
            observed_data = trainer.configurator(trainer.generative_model(num_observed_simulations))
        elif observed_data_type == "numpy":
            observed_data = trainer.configurator(trainer.generative_model(num_observed_simulations))['summary_conditions']

        MMD_sampling_distribution, MMD_observed = trainer.mmd_hypothesis_test(observed_data=observed_data,
                                                                              reference_data=reference_data,
                                                                              num_reference_simulations=num_reference_simulations,
                                                                              num_null_samples=num_null_samples,
                                                                              bootstrap=bootstrap)

        assert MMD_sampling_distribution.shape[0] == num_reference_simulations
        assert np.all(MMD_sampling_distribution > 0)
