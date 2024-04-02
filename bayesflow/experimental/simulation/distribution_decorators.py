
import functools
import keras

from bayesflow.experimental import utils
from bayesflow.experimental.simulation import distributions as D


class DistributionDecorator:
    conditional_distribution_type: type
    unconditional_distribution_type: type

    def __init__(self, is_batched: bool = False, is_conditional: bool = False):
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
        self.is_conditional = is_conditional

        if self.is_conditional:
            self.distribution_type = self.conditional_distribution_type
        else:
            self.distribution_type = self.unconditional_distribution_type

    def __call__(self, sample_fn: callable):
        if not self.is_batched:
            unbatched_sample_fn = sample_fn
            batched_sample_fn = keras.ops.vectorized_map(sample_fn)
        else:
            unbatched_sample_fn = sample_fn
            batched_sample_fn = sample_fn

        class Distribution(self.distribution_type):
            def sample(self, *args, **kwargs):
                return utils.apply_nested(keras.ops.convert_to_tensor, batched_sample_fn(*args, **kwargs))

            def __call__(self, *args, **kwargs):
                return unbatched_sample_fn(*args, **kwargs)

        instance = Distribution()

        functools.update_wrapper(instance, sample_fn)

        return instance


class PriorDecorator(DistributionDecorator):
    unconditional_distribution_type = D.UnconditionalPriorMixin
    conditional_distribution_type = D.ConditionalPriorMixin


class LikelihoodDecorator(DistributionDecorator):
    unconditional_distribution_type = D.UnconditionalLikelihoodMixin
    conditional_distribution_type = D.ConditionalLikelihoodMixin
