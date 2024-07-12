from functools import wraps
import keras
import numpy as np

from bayesflow.utils import filter_kwargs, stack_dicts

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

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, Tensor]:
        # try to use only valid keyword arguments
        kwargs = filter_kwargs(kwargs, self.sample_fn)

        sample_fn = self.sample_fn
        if self.is_numpy:
            sample_fn = self._convert_numpy(sample_fn)

        if not self.is_batched:
            sample_fn = self._convert_batched(sample_fn)

        return sample_fn(batch_shape, **kwargs)

    def _convert_batched(self, sample_fn: callable) -> callable:
        # use for loop, not vmap, because vmap does not preserve randomness for numpy
        @wraps(sample_fn)
        def wrapper(batch_shape, *args, **kwargs):
            batch_size = np.prod(batch_shape)
            data = []
            for i in range(batch_size):
                args_i = [arg[i] for arg in args]
                kwargs_i = {key: value[i] for key, value in kwargs.items()}
                data_i = sample_fn(*args_i, **kwargs_i)
                data.append(data_i)
            data = stack_dicts(data)

            return data

        return wrapper

    def _convert_numpy(self, sample_fn: callable) -> callable:
        @wraps(sample_fn)
        def wrapper(*args, **kwargs):
            # convert to numpy because numpy functions expect numpy arguments
            args = [keras.ops.convert_to_numpy(arg) for arg in args]
            kwargs = {key: keras.ops.convert_to_numpy(value) for key, value in kwargs.items()}
            data = sample_fn(*args, **kwargs)
            # convert to float32 to avoid 64-bit tensors on x64 systems (numpy default)
            return {key: keras.ops.convert_to_tensor(value, dtype="float32") for key, value in data.items()}

        return wrapper
