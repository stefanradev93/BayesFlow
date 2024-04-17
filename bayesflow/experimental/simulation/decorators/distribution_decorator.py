
import functools
import inspect
import keras

from functools import wraps

from bayesflow.experimental.types import Shape
from bayesflow.experimental.simulation.distributions import Distribution

from bayesflow.experimental import utils


class DistributionDecorator:
    def __new__(cls, *args, **kwargs):
        called_raw = len(args) == 1 and len(kwargs) == 0 and callable(args[0])

        if called_raw:
            # called as @decorator
            sample_fn = args[0]
            obj = super().__new__(cls)
            # use default args
            try:
                obj.__init__()
            except TypeError as e:
                raise TypeError(f"Decorator '{cls.__name__}' requires arguments and thus must be called as @{cls.__name__}(<args>)") from e
            return obj(sample_fn)

        # called as @decorator(*args, **kwargs)
        obj = super().__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def __init__(self, *, is_batched=False):
        self.is_batched = is_batched

    def __call__(self, sample_fn: callable) -> Distribution:
        if not self.is_batched:
            unbatched_sample_fn = sample_fn

            def batched_sample_fn(batch_shape, *args, **kwargs):
                if kwargs:
                    raise NotImplementedError("Keyword arguments are not yet supported.")

                # multi-argument function must take a list of tensors as single arg
                def f(tensors):
                    return sample_fn(*tensors[1:])

                batch_size = keras.ops.prod(batch_shape)
                dummy_argument = keras.ops.zeros((batch_size, 0))

                # draw flattened samples with dummy argument
                samples = keras.ops.vectorized_map(f, (dummy_argument, *args))

                # restore original batch shape
                samples = {key: keras.ops.reshape(val, batch_shape + keras.ops.shape(val)[1:]) for key, val in samples.items()}

                return samples
        else:
            def unbatched_sample_fn(*args, **kwargs):
                sample = sample_fn((1,), *args, **kwargs)
                return keras.ops.squeeze(sample, axis=0)

            batched_sample_fn = sample_fn

        class Instance(Distribution):
            _raw_sample_fn = sample_fn
            _unbatched_sample_fn = unbatched_sample_fn
            _batched_sample_fn = batched_sample_fn

            def __call__(self, *args, **kwargs):
                return unbatched_sample_fn(*args, **kwargs)

            def sample(self, batch_shape, *args, **kwargs):
                return batched_sample_fn(batch_shape, *args, **kwargs)

            def log_prob(self, *args, **kwargs):
                raise NotImplementedError("Density evaluation is not supported when using the distribution decorators.")

        instance = Instance()
        functools.update_wrapper(instance, sample_fn)
        return instance
