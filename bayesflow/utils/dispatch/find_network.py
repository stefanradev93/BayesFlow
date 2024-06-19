
import keras

from functools import singledispatch


@singledispatch
def find_network(arg, **kwargs):
    raise TypeError(f"Cannot infer network from {arg!r}.")


@find_network.register
def _(name: str, **kwargs):
    match name.lower():
        case "resnet":
            from bayesflow.networks import ResNet
            network = ResNet(**kwargs)
        case other:
            raise ValueError(f"Unsupported network name: '{other}'.")

    return network


@find_network.register
def _(network: keras.Layer):
    return network


@find_network.register
def _(constructor: type, **kwargs):
    return constructor(**kwargs)
