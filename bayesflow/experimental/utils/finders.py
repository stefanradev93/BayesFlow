
from functools import partial

import keras


def find_distribution(distribution: str | type, **kwargs):
    # TODO -> return type
    match distribution:
        case str() as name:
            match name.lower():
                case "normal":
                    from bayesflow.experimental.distributions import DiagonalNormal
                    distribution = DiagonalNormal(**kwargs)
                case other:
                    raise ValueError(f"Unsupported distribution name: '{other}'.")
        case type() as constructor:
            distribution = constructor(**kwargs)
        case other:
            raise TypeError(f"Cannot infer distribution from {other!r}.")

    return distribution


def find_network(network: str | keras.Layer | type, **kwargs) -> keras.Layer:
    match network:
        case str() as name:
            match name.lower():
                case "resnet":
                    from bayesflow.experimental.networks import ResNet
                    network = ResNet(**kwargs)
                case other:
                    raise ValueError(f"Unsupported network name: '{other}'.")
        case keras.Layer() as network:
            pass
        case type() as constructor:
            network = constructor(**kwargs)
        case other:
            raise TypeError(f"Cannot infer network from {other!r}.")

    return network


def find_pooling(pooling: str | keras.Layer | type, **kwargs) -> keras.Layer:
    match pooling:
        case str() as name:
            match name.lower():
                case "mean" | "avg":
                    pooling = keras.layers.Lambda(lambda inp: keras.ops.mean(inp, axis=-2))
                case "max":
                    pooling = keras.layers.Lambda(lambda inp: keras.ops.max(inp, axis=-2))
                case "min":
                    pooling = keras.layers.Lambda(lambda inp: keras.ops.min(inp, axis=-2))
                case "learnable" | "pma":
                    from bayesflow.experimental.networks.set_transformer.pma import PoolingByMultiheadAttention
                    pooling = PoolingByMultiheadAttention(**kwargs)
                case other:
                    raise ValueError(f"Unsupported pooling type: '{other}'.")
        case keras.Layer() as pooling:
            pass
        case type() as constructor:
            pooling = constructor(**kwargs)
        case other:
            raise TypeError(f"Cannot infer pooling type from {other!r}.")
    return pooling
