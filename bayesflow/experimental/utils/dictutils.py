
from bayesflow.experimental.types import Tensor
from keras import ops


def nested_getitem(data: dict, item: int) -> dict:
    """ Get the item-th element from a nested dictionary. """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = nested_getitem(value, item)
        else:
            result[key] = value[item]
    return result


def concatenate_tensors(tensor_dict: dict[str, Tensor], filter_list: list, axis: int = -1):
    """ Concatenates all tensors from tensor_dict using only keys from filter_list.
    An optional axis can be specified (default: last axis).
    """

    return ops.concatenate([v for k, v in tensor_dict.items() if k in filter_list], axis=axis)


def keras_kwargs(kwargs: dict):
    """ Keep dictionary keys that do not end with _kwargs. Used for propagating
    custom keyword arguments in custom models that inherit from keras.Model.
    """
    return {key: value for key, value in kwargs.items() if not key.endswith("_kwargs")}
