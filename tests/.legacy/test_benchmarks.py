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

import copy

import numpy as np
import pytest
import tensorflow as tf

from assets.benchmark_network_architectures import NETWORK_SETTINGS
from bayesflow import benchmarks
from bayesflow.amortizers import AmortizedLikelihood, AmortizedPosterior, AmortizedPosteriorLikelihood
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer


def _get_trainer_configuration(benchmark_name, mode):
    """Helper function to configure test ``Trainer`` instance."""

    # Clear tensorflow session
    tf.keras.backend.clear_session()

    # Setup benchmark instance
    benchmark = benchmarks.Benchmark(benchmark_name, mode=mode)

    # Setup posterior amortizer
    if mode == "posterior":
        amortizer = AmortizedPosterior(InvertibleNetwork(**NETWORK_SETTINGS[benchmark_name][mode]))
    elif mode == "likelihood":
        amortizer = AmortizedLikelihood(InvertibleNetwork(**NETWORK_SETTINGS[benchmark_name][mode]))
    else:
        amortizer = AmortizedPosteriorLikelihood(
            amortized_posterior=AmortizedPosterior(InvertibleNetwork(**NETWORK_SETTINGS[benchmark_name]["posterior"])),
            amortized_likelihood=AmortizedLikelihood(
                InvertibleNetwork(**NETWORK_SETTINGS[benchmark_name]["likelihood"])
            ),
        )
    trainer = Trainer(
        amortizer=amortizer,
        generative_model=benchmark.generative_model,
        learning_rate=0.0001,
        configurator=benchmark.configurator,
        memory=False,
    )
    return trainer


@pytest.mark.parametrize("benchmark_name", benchmarks.available_benchmarks)
@pytest.mark.parametrize("mode", ["posterior", "likelihood", "joint"])
def test_posterior(benchmark_name, mode):
    """This test will run posterior, likelihood, and joint estimation on all benchmarks. It will create a
    minimal ``Trainer`` instance and test whether the weights change after a couple of backpropagation updates.

    Implicitly, the function will test if the coupling ``GenerativeModel`` -> ``configurator`` ->
    ``Amortizer`` -> ``Trainer`` works.
    """

    # Default settings for testing
    epochs = 1
    iterations = 5
    batch_size = 16

    # Init trainer (including checks) and train
    trainer = _get_trainer_configuration(benchmark_name, mode=mode)
    trainable_variables_pre = copy.deepcopy(trainer.amortizer.trainable_variables)
    _ = trainer.train_online(epochs, iterations, batch_size)
    trainable_variables_post = copy.deepcopy(trainer.amortizer.trainable_variables)

    # Test whether weights change
    for before, after in zip(trainable_variables_pre, trainable_variables_post):
        assert np.any(before != after)
