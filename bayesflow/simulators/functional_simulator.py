from functools import wraps
import keras
import numpy as np

from bayesflow.utils import filter_kwargs, stack_dicts

from .simulator import Simulator
from ..types import Shape, Tensor


class FunctionalSimulator(Simulator):
    """Implements a functional simulator that can automatically detect batched/unbatched numpy/tensor output
    and convert accordingly.
    """

    def __init__(self, sample_fn: callable, *, is_batched: bool = None, is_numpy: bool = None):
        self.sample_fn = sample_fn
        self.is_batched = is_batched
        self.is_numpy = is_numpy

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, Tensor]:
        # try to use only valid keyword arguments
        kwargs = filter_kwargs(kwargs, self.sample_fn)

        if self.is_batched is None:
            self.is_batched = self._detect_is_batched(**kwargs)

        if self.is_numpy is None:
            self.is_numpy = self._detect_is_numpy(**kwargs)

        sample_fn = self.sample_fn
        if self.is_numpy:
            sample_fn = self._convert_numpy(sample_fn)

        if not self.is_batched:
            sample_fn = self._convert_batched(sample_fn)

        print(f"{sample_fn is self.sample_fn=}")

        return sample_fn(batch_shape, **kwargs)

    def _convert_batched(self, sample_fn: callable) -> callable:
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
            args = [keras.ops.convert_to_numpy(arg) for arg in args]
            kwargs = {key: keras.ops.convert_to_numpy(value) for key, value in kwargs.items()}
            data = sample_fn(*args, **kwargs)
            return {key: keras.ops.convert_to_tensor(value, dtype="float32") for key, value in data.items()}

        return wrapper

    def _detect_is_batched(self, **kwargs) -> bool:
        try:
            self.sample_fn((1,), **kwargs)
            return True
        except TypeError:
            # likely some argument error, check without batch shape
            try:
                kwargs = {key: value[0] for key, value in kwargs.items()}
                self.sample_fn(**kwargs)
                return False
            except Exception:
                # some other error, re-raise
                raise RuntimeError("Could not auto-detect batch status. Please pass it manually.")

    def _detect_is_numpy(self, **kwargs) -> bool:
        if self.is_batched:
            batch = self.sample_fn((1,), **kwargs)
        else:
            kwargs = {key: value[0] for key, value in kwargs.items()}
            batch = self.sample_fn(**kwargs)

        return not keras.ops.is_tensor(next(iter(batch.values())))
