import importlib
from functools import partial

from bayesflow.utils import batched_call


class Benchmark:
    def __init__(self, name: str, **kwargs):
        """
        Currently supported benchmarks:
        - bernoulli_glm
        - bernoulli_glm_raw
        - gaussian_linear
        - gaussian_linear_uniform
        - gaussian_mixture
        - inverse_kinematics
        - lotka_volterra
        - sir
        - slcp
        - slcp_distractors
        - two_moons

        TODO: Docs
        """

        self.name = name
        self.module = self.get_module(name)
        self.simulator = partial(getattr(self.module, "simulator"), **kwargs.pop("prior_kwargs", {}))

    def sample(self, batch_size: int):
        return batched_call(self.simulator, (batch_size,))

    @staticmethod
    def get_module(name):
        try:
            benchmark_module = importlib.import_module(f"bayesflow.benchmarks.{name}")
            return benchmark_module
        except Exception as error:
            raise ModuleNotFoundError(f"{name} is not a known benchmark") from error
