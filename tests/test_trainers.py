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
from pandas import DataFrame

import pytest

from bayesflow.networks import InvertibleNetwork, InvariantNetwork
from bayesflow.simulation import GenerativeModel
from bayesflow.trainers import Trainer
from bayesflow.amortizers import AmortizedPosterior, AmortizedLikelihood, AmortizedPosteriorLikelihood

def _prior(D=2, mu=0., sigma=1.0):
    """Helper minimal prior function."""
    return np.random.default_rng().normal(loc=mu, scale=sigma, size=D)

def _simulator(theta, n_obs=10, scale=1.0):
    """Helper minimal simulator function."""
    return np.random.default_rng().normal(loc=theta, scale=scale, size=(n_obs, theta.shape[0]))

def _create_training_setup(mode):
    """Helper function to create a relevant training setup."""
    
    # Create a generative model
    model = GenerativeModel(_prior, _simulator, name='test')

    # Case posterior inference
    if mode == 'posterior':
        summary_net = InvariantNetwork()
        inference_net = InvertibleNetwork(num_params=2, num_coupling_layers=2)
        amortizer = AmortizedPosterior(inference_net, summary_net)
    
    # Case likelihood inference
    elif mode == 'likelihood':
        surrogate_net = InvertibleNetwork(num_params=2, num_coupling_layers=2)
        amortizer = AmortizedLikelihood(surrogate_net)
    
    # Case joint inference
    else:
        summary_net = InvariantNetwork()
        inference_net = InvertibleNetwork(num_params=2, num_coupling_layers=2)
        p_amortizer = AmortizedPosterior(inference_net, summary_net)
        surrogate_net = InvertibleNetwork(num_params=2, num_coupling_layers=2)
        l_amortizer = AmortizedLikelihood(surrogate_net)
        amortizer = AmortizedPosteriorLikelihood(p_amortizer, l_amortizer)

    # Create and return trainer instance
    trainer = Trainer(generative_model=model, amortizer=amortizer)
    return trainer

@pytest.mark.parametrize("mode", ['posterior', 'likelihood'])
@pytest.mark.parametrize("reuse_optimizer", [True, False])
def test_train_online(mode, reuse_optimizer):
    """Tests the online training functionality."""

    # Create trainer and train online
    trainer = _create_training_setup(mode)
    h = trainer.train_online(
        epochs=2, 
        iterations_per_epoch=3, 
        batch_size=8, 
        reuse_optimizer=reuse_optimizer,
        skip_checks=True,
        memory=False,
    )

    # Assert (non)-existence of optimizer
    if reuse_optimizer:
        assert trainer.optimizer is not None
    else:
        assert trainer.optimizer is None

    # Assert type of history is data frame, meaning
    # losses were stored in the correct format
    assert type(h) is DataFrame


@pytest.mark.parametrize("mode", ['posterior', 'joint'])
@pytest.mark.parametrize("reuse_optimizer", [True, False])
def test_train_experience_replay(mode, reuse_optimizer):
    """Tests the experience replay training functionality."""

    # Create trainer and train with experience replay
    trainer = _create_training_setup(mode)
    h = trainer.train_experience_replay(
        epochs=3, 
        iterations_per_epoch=2, 
        batch_size=8, 
        reuse_optimizer=reuse_optimizer
    )

    # Assert (non)-existence of optimizer
    if reuse_optimizer:
        assert trainer.optimizer is not None
    else:
        assert trainer.optimizer is None

    # Assert type of history is data frame, meaning
    # losses were stored in the correct format
    assert type(h) is DataFrame


@pytest.mark.parametrize("mode", ['likelihood', 'joint'])
@pytest.mark.parametrize("reuse_optimizer", [True, False])
def test_train_offline(mode, reuse_optimizer):
    """Tests the offline training functionality."""

    # Create trainer and data and train offline
    trainer = _create_training_setup(mode)
    simulations = trainer.generative_model(100)
    h = trainer.train_offline(
        simulations_dict=simulations,
        epochs=2, 
        batch_size=16, 
        reuse_optimizer=reuse_optimizer,
    )

    # Assert (non)-existence of optimizer
    if reuse_optimizer:
        assert trainer.optimizer is not None
    else:
        assert trainer.optimizer is None

    # Assert type of history is data frame, meaning
    # losses were stored in the correct format
    assert type(h) is DataFrame


@pytest.mark.parametrize("mode", ['likelihood', 'posterior'])
@pytest.mark.parametrize("reuse_optimizer", [True, False])
def test_train_rounds(mode, reuse_optimizer):
    """Tests the offline training functionality."""

    # Create trainer and data and train offline
    trainer = _create_training_setup(mode)
    h = trainer.train_rounds(
        rounds=2,
        sim_per_round=32,
        epochs=2, 
        batch_size=8, 
        reuse_optimizer=reuse_optimizer,
    )

    # Assert (non)-existence of optimizer
    if reuse_optimizer:
        assert trainer.optimizer is not None
    else:
        assert trainer.optimizer is None

    # Assert type of history is data frame, meaning
    # losses were stored in the correct format
    assert type(h) is DataFrame
