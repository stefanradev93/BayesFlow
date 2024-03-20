
import functools
import keras

from bayesflow.experimental.backend_agnostic import utils
from bayesflow.experimental.backend_agnostic.simulation.distributions import ContextDistributionMixin, ObservableDistributionMixin, ParameterDistributionMixin


class DistributionDecorator:
    distribution_type: type

    def __init__(self, is_batched: bool = False):
        if callable(is_batched):
            message = f"""
            The decorator {self.__class__.__name__} accepts arguments and thus must be used with parentheses.
            Example:
            >>> @{self.__class__.__name__}():  <--- Parentheses here
            >>> def f(...):
            >>>     pass
            """
            raise RuntimeError(message)

        self.is_batched = is_batched

    def __call__(self, sample_fn: callable):
        if not self.is_batched:
            sample_fn = keras.ops.vectorized_map(sample_fn)

        class Distribution(self.distribution_type):
            def sample(self, *args, **kwargs):
                return utils.apply_nested(keras.ops.convert_to_tensor, sample_fn(*args, **kwargs))

            def __call__(self, *args, **kwargs):
                return self.sample(*args, **kwargs)

        instance = Distribution()

        functools.update_wrapper(instance, sample_fn)

        return instance


class ContextPriorDecorator(DistributionDecorator):
    distribution_type = ContextPriorMixin


class ObservableDistributionDecorator(DistributionDecorator):
    distribution_type = ObservableDistributionMixin


class ParameterDistributionDecorator(DistributionDecorator):
    distribution_type = ParameterDistributionMixin
