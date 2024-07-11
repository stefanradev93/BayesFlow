import inspect
import logging
from collections.abc import Sequence

import keras
import numpy as np
from keras import ops

from bayesflow.types import Shape, Tensor


def convert_kwargs(f, *args, **kwargs) -> dict[str, any]:
    """Convert positional and keyword arguments to just keyword arguments for f"""
    if not args:
        return kwargs

    signature = inspect.signature(f)

    parameters = dict(zip(signature.parameters, args))

    for name, value in kwargs.items():
        if name in parameters:
            raise TypeError(f"{f.__name__}() got multiple arguments for argument '{name}'")

        parameters[name] = value

    return parameters


def convert_args(f, *args, **kwargs) -> tuple[any, ...]:
    """Convert positional and keyword arguments to just positional arguments for f"""
    if not kwargs:
        return args

    signature = inspect.signature(f)

    # convert to just kwargs first
    kwargs = convert_kwargs(f, *args, **kwargs)

    parameters = []
    for name, param in signature.parameters.items():
        if param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]:
            continue

        parameters.append(kwargs.get(name, param.default))

    return tuple(parameters)


def batched_call(f: callable, batch_shape: Shape, *args: Tensor, **kwargs: Tensor):
    """Call f, automatically vectorizing to batch_shape if required.

    :param f: The function to call.
        May accept any number of tensor or numpy array arguments.
        Must return a dictionary of tensors or numpy arrays.

    :param batch_shape: The shape of the batch. If f is not already batched, it will be called
        prod(batch_shape) times.

    :param args: Positional arguments to f

    :param kwargs: Keyword arguments to f

    :return: A dictionary of batched tensors or numpy arrays.
    """
    try:
        # already batched
        data = f(batch_shape, *args, **kwargs)

        # convert numpy to keras
        data = {key: keras.ops.convert_to_tensor(value) for key, value in data.items()}
    except TypeError:
        # for loop fallback
        batch_size = np.prod(batch_shape)
        data = []
        for b in range(batch_size):
            # get args and kwargs for this index
            args_i = [args[i][b] for i in range(len(args))]
            kwargs_i = {k: v[b] for k, v in kwargs.items()}

            data_i = f(*args_i, **kwargs_i)

            # convert numpy to keras
            data_i = {key: keras.ops.convert_to_tensor(value) for key, value in data_i.items()}

            data.append(data_i)

        data = stack_dicts(data, axis=0)

        # reshape to batch_shape
        data = {key: keras.ops.reshape(value, batch_shape + keras.ops.shape(value)[1:]) for key, value in data.items()}

    return data


def filter_kwargs(kwargs: dict[str, any], f: callable) -> dict[str, any]:
    """Filter keyword arguments for f"""
    signature = inspect.signature(f)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

    return kwargs


def filter_concatenate(data: dict[str, Tensor], keys: Sequence[str], axis: int = -1) -> Tensor | None:
    """Filters and then concatenates all tensors from data using only keys from the given sequence.
    An optional axis can be specified (default: last axis).
    """
    if not keys:
        return None

    # ensure every key is present
    tensors = [data[key] for key in keys]

    try:
        return keras.ops.concatenate(tensors, axis=axis)
    except ValueError as e:
        shapes = [t.shape for t in tensors]
        raise ValueError(f"Cannot trivially concatenate tensors {keys} with shapes {shapes}") from e


def filter_tuple(data: dict[str, Tensor], keys: Sequence[str]) -> tuple[Tensor, ...]:
    """Filters all tensors from data using only keys from the given sequence and returns them as a tuple."""
    if not keys:
        return ()

    return tuple(data[key] for key in keys)


def keras_kwargs(kwargs: dict) -> dict:
    """Keep dictionary keys that do not end with _kwargs. Used for propagating
    custom keyword arguments in custom models that inherit from keras.Model.
    """
    return {key: value for key, value in kwargs.items() if not key.endswith("_kwargs")}


def concatenate_dicts(data: list[dict[str, Tensor]], axis: int = -1) -> dict[str, Tensor]:
    """Concatenates tensors in multiple dictionaries into a single dictionary."""
    if not all([d.keys() == data[0].keys() for d in data]):
        raise ValueError("Dictionaries must have the same keys.")

    result = {}

    for key in data[0].keys():
        result[key] = keras.ops.concatenate([d[key] for d in data], axis=axis)

    return result


def stack_dicts(data: list[dict[str, Tensor]], axis: int = 0) -> dict[str, Tensor]:
    """Stacks tensors in multiple dictionaries into a single dictionary."""
    if not all([d.keys() == data[0].keys() for d in data]):
        raise ValueError("Dictionaries must have the same keys.")

    result = {}

    for key in data[0].keys():
        result[key] = keras.ops.stack([d[key] for d in data], axis=axis)

    return result


def process_output(outputs: dict[str, Tensor], convert_to_numpy: bool = True) -> dict[str, Tensor]:
    """Utility function to apply common post-processing steps to the outputs of an approximator."""

    # Remove trailing first axis for single data sets
    outputs = {k: ops.squeeze(v, axis=0) if ops.shape(v)[0] == 1 else v for k, v in outputs.items()}

    # Warn if any NaNs present in output
    for k, v in outputs.items():
        nan_mask = ops.isnan(v)
        if ops.any(nan_mask):
            logging.warning(f"A total of {ops.sum(nan_mask)} NaN values found for output {k}.")

    # Warn if any inf present in output
    for k, v in outputs.items():
        inf_mask = ops.isinf(v)
        if ops.any(inf_mask):
            logging.warning(f"A total of {ops.sum(inf_mask)} inf values found for output {k}.")

    if convert_to_numpy:
        outputs = {k: ops.convert_to_numpy(v) for k, v in outputs.items()}
    return outputs
