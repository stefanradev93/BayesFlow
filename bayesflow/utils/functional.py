from collections.abc import Mapping, Sequence
import keras
import numpy as np

from bayesflow.types import Shape


def batched_call(
    f: callable, batch_shape: Shape, args: Sequence[any], kwargs: Mapping[str, any], flatten: bool = False
) -> list:
    """Map f over the given batch shape with a for loop, preserving randomness unlike the keras built-in map apis.

    :param f: The function to call.
        May accept any number of tensor or non-tensor arguments.
        Tensor arguments are indexed by the first len(batch_shape) axes.
        Non-tensor arguments are passed as-is.

    :param batch_shape: The shape of the batch.

    :param args: Positional arguments to f

    :param kwargs: Keyword arguments to f

    :param flatten: Whether to flatten the output list.

    :return: A list of outputs of f for each element in the batch.
    """
    # get a flat list of index tuples
    indices = np.indices(batch_shape)
    indices = np.reshape(indices, (len(batch_shape), -1)).T
    indices = indices.tolist()
    indices = [tuple(index) for index in indices]

    # initialize nested output list
    outputs = np.empty(batch_shape, dtype="object")

    for index in indices:
        positional_args = [arg[index] if keras.ops.is_tensor(arg) else arg for arg in args]
        keyword_args = {key: value[index] if keras.ops.is_tensor(value) else value for key, value in kwargs.items()}

        outputs[*index] = f(*positional_args, **keyword_args)

    if flatten:
        outputs = outputs.flatten()

    return outputs.tolist()
