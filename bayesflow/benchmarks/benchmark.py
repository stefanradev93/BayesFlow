import importlib
from functools import partial

from bayesflow.simulators import SequentialSimulator


class Benchmark(SequentialSimulator):
    def __init__(self, name: str, **kwargs):
        """#TODO"""

        self.name = name
        self.module = self.get_module(name)

        prior = partial(getattr(self.module, "prior"), **kwargs.pop("prior_kwargs", {}))
        obs_model = partial(getattr(self.module, "observation_model"), **kwargs.pop("observation_model_kwargs", {}))
        super().__init__([prior, obs_model])

    @staticmethod
    def get_module(name):
        """Loads the corresponding benchmark file under bayesflow.benchmarks.<benchmark_name> as a
        module and returns it.
        """

        try:
            benchmark_module = importlib.import_module(f"bayesflow.benchmarks.{name}")
            return benchmark_module
        except Exception as error:
            raise ModuleNotFoundError(f"{name} is not a known benchmark") from error
