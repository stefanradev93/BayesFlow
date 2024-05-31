
def nested_getitem(data: dict, item: int) -> dict:
    """ Get the item-th element from a nested dictionary """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = nested_getitem(value, item)
        else:
            result[key] = value[item]
    return result
