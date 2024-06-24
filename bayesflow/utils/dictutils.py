import inspect
import keras

from collections.abc import Sequence

from bayesflow.types import Tensor


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


def batched_call(f, batch_size, *args, **kwargs):
    """Call f, automatically vectorizing to batch_size if required"""
    try:
        data = f((batch_size,), *args, **kwargs)
        data = {key: keras.ops.convert_to_tensor(value) for key, value in data.items()}
        return data
    except TypeError:
        pass

    # no way to get both randomness and support for numpy sampling without a for loop :(
    data = [f(*args, **kwargs) for _ in range(batch_size)]

    data_dict = {}
    for key in data[0].keys():
        # gather tensors for key into list
        tensors = [data[i][key] for i in range(len(data))]
        data_dict[key] = keras.ops.stack(tensors, axis=0)

    return data_dict


def filter_concatenate(data: dict[str, Tensor], keys: Sequence[str], axis: int = -1) -> Tensor:
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


def keras_kwargs(kwargs: dict):
    """Keep dictionary keys that do not end with _kwargs. Used for propagating
    custom keyword arguments in custom models that inherit from keras.Model.
    """
    return {key: value for key, value in kwargs.items() if not key.endswith("_kwargs")}
