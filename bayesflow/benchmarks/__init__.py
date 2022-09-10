# Copyright 2022 The BayesFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module implements all 10 benchmark models (tasks) from the paper:
#
# Lueckmann, J. M., Boelts, J., Greenberg, D., Goncalves, P., & Macke, J. (2021).
# Benchmarking simulation-based inference.
# In International Conference on Artificial Intelligence and Statistics (pp. 343-351). PMLR.
#
# https://arxiv.org/pdf/2101.04653.pdf
#
# However, it lifts the dependency on `torch` and implements the models as ready-made
# tuples of prior and simulator functions capable of interacting with BayesFlow.
# Note: All default hyperparameters are set according to the paper.

import importlib

from bayesflow.forward_inference import GenerativeModel, Prior
from bayesflow.exceptions import ConfigurationError

available_benchmarks = [
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "lotka_volterra",
    "sir",
    "slcp",
    "slcp_distractors",
    "two_moons"
]


def get_benchmark_module(benchmark_name):
    """ Loads the corresponding benchmark file under bayesflow.benchmarks.<benchmark_name> as a
    module and returns it.
    """

    try:
        benchmark_module = importlib.import_module(f'bayesflow.benchmarks.{benchmark_name}')
        return benchmark_module
    except ModuleNotFoundError:
        raise ConfigurationError(f"You need to provide a valid name from: {available_benchmarks}")


class Benchmark:
    """TODO"""
    def __init__(self, benchmark_name):

        self.benchmark_name = benchmark_name
        self.benchmark_module = get_benchmark_module(self.benchmark_name)
        self.benchmark_info = getattr(self.benchmark_module, 'bayesflow_benchmark_info')
        self.generative_model = GenerativeModel(
            prior=Prior(
                prior_fun=getattr(self.benchmark_module, 'prior'),
                param_names=self.benchmark_info['parameter_names']
            ),
            simulator=getattr(self.benchmark_module, 'simulator'),
            simulator_is_batched=self.benchmark_info['simulator_is_batched']
        )

        self.configurator = getattr(self.benchmark_module, 'configurator')
