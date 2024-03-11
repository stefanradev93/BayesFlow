
def nested_getitem(data: dict, item: int) -> dict:
    """ Get the item-th element from a nested dictionary """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = nested_getitem(value, item)
        else:
            result[key] = value[item]
    return result


def nested_merge(a: dict, b: dict) -> dict:
    """ Merge a nested dictionary A into another nested dictionary B """
    for key, value in a.items():
        if isinstance(value, dict):
            b[key] = nested_merge(value, b.get(key, {}))
        else:
            b[key] = value
    return b
