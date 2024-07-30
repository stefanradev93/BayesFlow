import inspect
import keras

from collections.abc import Mapping

from bayesflow.types import Tensor

from . import logging


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
    outputs = {k: keras.ops.squeeze(v, axis=0) if keras.ops.shape(v)[0] == 1 else v for k, v in outputs.items()}

    # Warn if any NaNs present in output
    for k, v in outputs.items():
        nan_mask = keras.ops.isnan(v)
        if keras.ops.any(nan_mask):
            logging.warning("Found a total of {n:d} nan values for output {k}.", n=int(keras.ops.sum(nan_mask)), k=k)

    # Warn if any inf present in output
    for k, v in outputs.items():
        inf_mask = keras.ops.isinf(v)
        if keras.ops.any(inf_mask):
            logging.warning("Found a total of {n:d} inf values for output {k}.", n=int(keras.ops.sum(inf_mask)), k=k)

    if convert_to_numpy:
        outputs = {k: keras.ops.convert_to_numpy(v) for k, v in outputs.items()}
    return outputs
