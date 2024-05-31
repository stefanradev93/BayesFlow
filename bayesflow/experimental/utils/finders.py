from typing import Callable

import keras

import bayesflow.experimental.distributions as D
import bayesflow.experimental.networks as N


def find_distribution(distribution: str | D.Distribution | Callable, **kwargs) -> D.Distribution:
    match distribution:
        case str() as name:
            match name.lower():
                case "normal":
                    distribution = D.Normal()
                case other:
                    raise NotImplementedError(f"Unsupported distribution name: '{other}'.")
        case D.Distribution() as distribution:
            pass
        case Callable() as constructor:
            distribution = constructor(**kwargs)
        case other:
            raise TypeError(f"Cannot infer distribution from {other!r}.")

    return distribution


def find_network(network: str | keras.Layer | Callable, **kwargs) -> keras.Layer:
    match network:
        case str() as name:
            match name.lower():
                case "resnet":
                    network = N.ResNet(**kwargs)
                case other:
                    raise NotImplementedError(f"Unsupported network name: '{other}'.")
        case keras.Layer() as network:
            pass
        case Callable() as constructor:
            network = constructor(**kwargs)
        case other:
            raise TypeError(f"Cannot infer network from {other!r}.")

    return network
