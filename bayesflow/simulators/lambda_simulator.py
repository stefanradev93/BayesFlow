from collections.abc import Mapping
import numpy as np

from bayesflow.utils import batched_call, filter_kwargs, tree_stack

from .simulator import Simulator
from ..types import Shape


class LambdaSimulator(Simulator):
    """Implements a simulator based on a lambda function.
    Can automatically convert unbatched into batched output.
    """

    def __init__(self, sample_fn: callable, *, is_batched: bool = False, cast_dtypes: Mapping[str, str] = None):
        self.sample_fn = sample_fn
        self.is_batched = is_batched

        if cast_dtypes is None:
            cast_dtypes = {"float64": "float32"}

        self.cast_dtypes = cast_dtypes

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        # try to use only valid keyword arguments
        kwargs = filter_kwargs(kwargs, self.sample_fn)

        if self.is_batched:
            data = self.sample_fn(batch_shape, **kwargs)
        else:
            data = batched_call(self.sample_fn, batch_shape, args=(), kwargs=kwargs, flatten=True)
            data = tree_stack(data, axis=0)

            # restore batch shape
            data = {key: np.reshape(value, batch_shape + np.shape(value)[1:]) for key, value in data.items()}

        for key, value in data.items():
            dtype = str(value.dtype)
            if dtype in self.cast_dtypes:
                data[key] = value.astype(self.cast_dtypes[dtype])

        return data
