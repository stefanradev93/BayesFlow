
from keras import ops

from typing import Sequence

from bayesflow.experimental.types import Tensor


def nested_getitem(data: dict, item: int) -> dict:
    """ Get the item-th element from a nested dictionary. """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = nested_getitem(value, item)
        else:
            result[key] = value[item]
    return result


def filter_concatenate(data: dict[str, Tensor], keys: Sequence[str], axis: int = -1) -> Tensor:
    """ Filters and then concatenates all tensors from data using only keys from the given sequence.
    An optional axis can be specified (default: last axis).
    """
    if not keys:
        return None

    # ensure every key is present
    tensors = [data[key] for key in keys]

    try:
        return ops.concatenate(tensors, axis=axis)
    except ValueError as e:
        shapes = [t.shape for t in tensors]
        raise ValueError(f"Cannot trivially concatenate tensors {keys} with shapes {shapes}") from e


def keras_kwargs(kwargs: dict):
    """ Keep dictionary keys that do not end with _kwargs. Used for propagating
    custom keyword arguments in custom models that inherit from keras.Model.
    """
    return {key: value for key, value in kwargs.items() if not key.endswith("_kwargs")}
