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

        match self.is_batched, self.is_numpy:
            case (True, False):
                # output is already batched and uses tensors: we don't have to do anything
                data = self.sample_fn(batch_shape, **kwargs)
            case (False, False):
                # output uses tensors, but is not batched: convert output to batched
                # note: we have to use a for loop instead of the vmap api to preserve randomness
                tensor_kwargs = {key: value for key, value in kwargs.items() if keras.ops.is_tensor(value)}
                other_kwargs = {key: value for key, value in kwargs.items() if key not in tensor_kwargs}

                # get a flat list of index tuples
                indices = np.indices(batch_shape)
                indices = np.reshape(indices, (len(batch_shape), -1)).T
                indices = indices.tolist()
                indices = [tuple(index) for index in indices]

                data = []

                for index in indices:
                    kwargs = {key: value[index] for key, value in tensor_kwargs.items()} | other_kwargs
                    data.append(self.sample_fn(**kwargs))

                data = stack_dicts(data)

                # restore batch shape
                data = {
                    key: keras.ops.reshape(value, batch_shape + keras.ops.shape(value)[1:])
                    for key, value in data.items()
                }
            case (True, True):
                # output is batched, but does not use tensors: convert input/output
                kwargs = {key: keras.ops.convert_to_numpy(value) for key, value in kwargs.items()}
                data = self.sample_fn(batch_shape, **kwargs)
                data = {key: keras.ops.convert_to_tensor(value, dtype="float32") for key, value in data.items()}
            case (False, True):
                # output is neither batched nor using tensors: do everything
                # note: This is very similar to the other batched=False case.
                # We do not apply DRY here since we only repeat ourselves once.
                # Split this into functions if it comes up more often.
                tensor_kwargs = {
                    key: keras.ops.convert_to_numpy(value)
                    for key, value in kwargs.items()
                    if keras.ops.is_tensor(value)
                }
                other_kwargs = {key: value for key, value in kwargs.items() if key not in tensor_kwargs}

                # get a flat list of index tuples
                indices = np.indices(batch_shape)
                indices = np.reshape(indices, (len(batch_shape), -1)).T
                indices = indices.tolist()
                indices = [tuple(index) for index in indices]

                data = []
                for index in indices:
                    kwargs = {key: value[index] for key, value in tensor_kwargs.items()} | other_kwargs
                    data.append(self.sample_fn(**kwargs))

                data = stack_dicts(data)

                # convert to tensors
                data = {key: keras.ops.convert_to_tensor(value, dtype="float32") for key, value in data.items()}

                # restore batch shape
                data = {
                    key: keras.ops.reshape(value, batch_shape + keras.ops.shape(value)[1:])
                    for key, value in data.items()
                }
            case _:
                # not reachable
                raise RuntimeError(
                    f"Unexpected value for is_batched ({self.is_batched}) or is_numpy ({self.is_numpy})."
                )

        return data
