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

# This module implements all 10 benchmark models (tasks) from the paper:
#
# Lueckmann, J. M., Boelts, J., Greenberg, D., Goncalves, P., & Macke, J. (2021).
# Benchmarking simulation-based inference.
# In International Conference on Artificial Intelligence and Statistics (pp. 343-351). PMLR.
#
# https://arxiv.org/pdf/2101.04653.pdf
#
# However, it lifts the dependency on `PyTorch` and implements the models as ready-made
# tuples of prior and simulator functions capable of interacting with BayesFlow.
# Note: All default hyperparameters are set according to the paper.

import importlib
from functools import partial

import numpy as np

from bayesflow.exceptions import ConfigurationError
from bayesflow.simulation import GenerativeModel, Prior

available_benchmarks = [
    "gaussian_linear",
    "gaussian_linear_uniform",
    "slcp",
    "slcp_distractors",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "gaussian_mixture",
    "two_moons",
    "sir",
    "lotka_volterra",
    "inverse_kinematics",
]


def get_benchmark_module(benchmark_name):
    """Loads the corresponding benchmark file under bayesflow.benchmarks.<benchmark_name> as a
    module and returns it.
    """

    try:
        benchmark_module = importlib.import_module(f"bayesflow.benchmarks.{benchmark_name}")
        return benchmark_module
    except ModuleNotFoundError:
        raise ConfigurationError(f"You need to provide a valid name from: {available_benchmarks}")


class Benchmark:
    """Interface class for a benchmark."""

    def __init__(self, name, mode="joint", seed=None, **kwargs):
        """Creates a benchmark generative model by using the blueprint contained
        in a benchmark file.

        Parameters
        ----------
        name : str
            The name of the benchmark file (without suffix, i.e., .py) to use as a blueprint.
        mode           : str, otpional, default: 'joint'
            The mode in which to configure the data, should be in ('joint', 'posterior', 'likelihood')
        seed           : int or None, optional, default: None
            The seed to use if reproducibility is required. Will be passed to a numpy RNG.
        **kwargs       : dict
            Optional keyword arguments.

            If 'sim_kwargs' is present, key-value pairs will be interpreted as arguments for the simulator
            and propagated accordingly.

            If 'prior_kwargs' is present, key-value pairs will be interpreted as arguments for the prior
            and propagated accordingly.
        """

        self.benchmark_name = name
        self._rng = np.random.default_rng(seed)
        self.benchmark_module = get_benchmark_module(self.benchmark_name)
        self.benchmark_info = getattr(self.benchmark_module, "bayesflow_benchmark_info")

        # Prepare partial simulator function with optional keyword arguments
        if kwargs.get("sim_kwargs") is not None:
            _simulator = partial(
                getattr(self.benchmark_module, "simulator"), rng=self._rng, **kwargs.pop("sim_kwargs", {})
            )
        else:
            _simulator = partial(getattr(self.benchmark_module, "simulator"), rng=self._rng)

        # Prepare partial prior function with optional keyword arguments
        if kwargs.get("prior_kwargs") is not None:
            _prior = partial(getattr(self.benchmark_module, "prior"), rng=self._rng, **kwargs.pop("prior_kwargs", {}))
        else:
            _prior = partial(getattr(self.benchmark_module, "prior"), rng=self._rng)

        # Prepare generative model
        self.generative_model = GenerativeModel(
            prior=Prior(
                prior_fun=_prior,
                param_names=self.benchmark_info["parameter_names"],
            ),
            simulator=_simulator,
            simulator_is_batched=self.benchmark_info["simulator_is_batched"],
            name=self.benchmark_name,
        )
        self.configurator = getattr(self.benchmark_module, "configurator")
        self.configurator = partial(self.configurator, mode=mode)
