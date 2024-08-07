from collections.abc import Mapping
import numpy as np

from bayesflow.utils import filter_kwargs, tree_stack

from .simulator import Simulator
from ..types import Shape


class LambdaSimulator(Simulator):
    def __init__(self, sample_fn: callable, *, is_batched: bool = False, cast_dtypes: Mapping[str, str] = "default"):
        """Implements a simulator based on a (batched or unbatched) sampling function.
        Outputs will always be in batched format.
        :param sample_fn: The sampling function.
            If in batched format, must accept a batch_shape argument as the first positional argument.
            If in unbatched format (the default), may accept any keyword arguments.
            Must return a dictionary of string keys and numpy array values.
        :param is_batched: Whether the sampling function is in batched format.
        :param cast_dtypes: Output data types to cast to.
            By default, we convert float64 (the default for numpy on x64 systems)
            to float32 (the default for deep learning on any system).
        """
        self.sample_fn = sample_fn
        self.is_batched = is_batched

        if cast_dtypes == "default":
            cast_dtypes = {"float64": "float32"}

        self.cast_dtypes = cast_dtypes

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        # try to use only valid keyword arguments
        kwargs = filter_kwargs(kwargs, self.sample_fn)

        if self.is_batched:
            data = self.sample_fn(batch_shape, **kwargs)
        else:
            data = np.empty(batch_shape, dtype="object")

            for index in np.ndindex(batch_shape):
                index_kwargs = {
                    key: value[index] if isinstance(value, np.ndarray) else value for key, value in kwargs.items()
                }
                data[index] = self.sample_fn(**index_kwargs)

            data = data.flatten().tolist()

            # convert 0D float outputs to 0D numpy arrays
            for i in range(len(data)):
                data[i] = {key: np.array(value) for key, value in data[i].items()}

            data = tree_stack(data, axis=0, numpy=True)

            # restore batch shape
            data = {key: np.reshape(value, batch_shape + np.shape(value)[1:]) for key, value in data.items()}

        for key, value in data.items():
            dtype = str(value.dtype)
            if dtype in self.cast_dtypes:
                data[key] = value.astype(self.cast_dtypes[dtype])

        return data
