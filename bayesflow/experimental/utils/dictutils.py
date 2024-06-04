
def nested_getitem(data: dict, item: int) -> dict:
    """ Get the item-th element from a nested dictionary """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = nested_getitem(value, item)
        else:
            result[key] = value[item]
    return result


def keras_kwargs(kwargs: dict):
    """ Keep dictionary keys that do not end with _kwargs. Used for propagating
    custom keyword arguments in custom models that inherit from keras.Model.
    """
    return {key: value for key, value in kwargs.items() if not key.endswith("_kwargs")}
