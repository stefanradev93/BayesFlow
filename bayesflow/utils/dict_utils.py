import inspect
import keras
from keras import ops

from collections.abc import Mapping, Sequence

from bayesflow.types import Tensor

from . import logging


def concatenate_dicts(data: list[Mapping[str, Tensor]], axis: int = -1) -> Mapping[str, Tensor]:
    """Concatenates tensors in multiple dictionaries into a single dictionary."""
    if not all([d.keys() == data[0].keys() for d in data]):
        raise ValueError("Dictionaries must have the same keys.")

    result = {}

    for key in data[0].keys():
        result[key] = keras.ops.concatenate([d[key] for d in data], axis=axis)

    return result


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


def convert_kwargs(f, *args, **kwargs) -> Mapping[str, any]:
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


def filter_concatenate(data: Mapping[str, Tensor], keys: Sequence[str], axis: int = -1) -> Tensor | None:
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


def filter_kwargs(kwargs: Mapping[str, any], f: callable) -> Mapping[str, any]:
    """Filter keyword arguments for f"""
    signature = inspect.signature(f)

    if inspect.Parameter.VAR_KEYWORD in signature.parameters:
        # the signature has **kwargs
        return kwargs

    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

    return kwargs


def keras_kwargs(kwargs: Mapping) -> Mapping:
    """Keep dictionary keys that do not end with _kwargs. Used for propagating
    custom keyword arguments in custom models that inherit from keras.Model.
    """
    return {key: value for key, value in kwargs.items() if not key.endswith("_kwargs")}


def process_output(outputs: Mapping[str, Tensor], convert_to_numpy: bool = True) -> Mapping[str, Tensor]:
    """Utility function to apply common post-processing steps to the outputs of an approximator."""

    # Remove trailing first axis for single data sets
    outputs = {k: ops.squeeze(v, axis=0) if ops.shape(v)[0] == 1 else v for k, v in outputs.items()}

    # Warn if any NaNs present in output
    for k, v in outputs.items():
        nan_mask = ops.isnan(v)
        if ops.any(nan_mask):
            logging.warning("Found a total of {n:d} nan values for output {k}.", n=int(ops.sum(nan_mask)), k=k)

    # Warn if any inf present in output
    for k, v in outputs.items():
        inf_mask = ops.isinf(v)
        if ops.any(inf_mask):
            logging.warning("Found a total of {n:d} inf values for output {k}.", n=int(ops.sum(inf_mask)), k=k)

    if convert_to_numpy:
        outputs = {k: ops.convert_to_numpy(v) for k, v in outputs.items()}
    return outputs


def stack_dicts(data: list[Mapping[str, Tensor]], axis: int = 0) -> Mapping[str, Tensor]:
    """Stacks tensors in multiple dictionaries into a single dictionary."""
    if not all([d.keys() == data[0].keys() for d in data]):
        raise ValueError("Dictionaries must have the same keys.")

    result = {}

    for key in data[0].keys():
        result[key] = keras.ops.stack([d[key] for d in data], axis=axis)

    return result
