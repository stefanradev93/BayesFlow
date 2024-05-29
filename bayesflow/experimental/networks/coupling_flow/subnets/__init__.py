
from typing import Callable

import keras

import bayesflow.experimental.networks as networks


def find_subnet(subnet: str | keras.Layer | Callable, **kwargs) -> keras.Layer:
    """ Find subnetworks by name and configure them to use lazy in- and output dimensions. """
    match subnet:
        case str() as name:
            match name.lower():
                case "resnet":
                    return networks.ResNet(**kwargs)
                case other:
                    raise NotImplementedError(f"Unsupported subnet name: '{other}'.")
        case keras.Layer() as layer:
            return layer
        case callable() as constructor:
            return constructor(**kwargs)
        case other:
            raise NotImplementedError(f"Cannot infer subnet from {other!r}.")
