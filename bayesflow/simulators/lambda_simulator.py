from functools import wraps
import keras

from bayesflow.utils import batched_call, filter_kwargs, stack_dicts

from .simulator import Simulator
from ..types import Shape, Tensor


class LambdaSimulator(Simulator):
    """Implements a simulator based on a lambda function.
    Can automatically convert unbatched into batched and numpy into keras output.
    """

    def __init__(self, sample_fn: callable, *, is_batched: bool = False, is_numpy: bool = True):
        self.sample_fn = sample_fn
        self.is_batched = is_batched
        self.is_numpy = is_numpy

    def sample(self, batch_shape: Shape, *, numpy: bool = False, **kwargs) -> dict[str, Tensor]:
        # try to use only valid keyword arguments
        kwargs = filter_kwargs(kwargs, self.sample_fn)

        if self.is_numpy:
            sample_fn = self._wrap_numpy(self.sample_fn)
        else:
            sample_fn = self.sample_fn

        if self.is_batched:
            data = sample_fn(batch_shape, **kwargs)
        else:
            data = batched_call(sample_fn, batch_shape, args=(), kwargs=kwargs, flatten=True)
            data = stack_dicts(data, axis=0)

            # restore batch shape
            data = {
                key: keras.ops.reshape(value, batch_shape + keras.ops.shape(value)[1:]) for key, value in data.items()
            }

        if numpy:
            data = {key: keras.ops.convert_to_numpy(value) for key, value in data.items()}

        return data

    @staticmethod
    def _wrap_numpy(f: callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            args = [keras.ops.convert_to_numpy(arg) if keras.ops.is_tensor(arg) else arg for arg in args]
            kwargs = {
                key: keras.ops.convert_to_numpy(value) if keras.ops.is_tensor(value) else value
                for key, value in kwargs.items()
            }
            samples = f(*args, **kwargs)
            samples = {key: keras.ops.convert_to_tensor(value, dtype="float32") for key, value in samples.items()}
            return samples

        return wrapper
