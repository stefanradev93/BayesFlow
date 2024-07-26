from functools import singledispatch
import keras

from bayesflow.types import Tensor
from bayesflow.types.tensor import BackendTensor


@singledispatch
def find_cost(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer cost matrix from {arg!r}.")


@find_cost.register
def _(name: str, x1: Tensor, x2: Tensor, **kwargs):
    n = keras.ops.shape(x1)[0]
    m = keras.ops.shape(x2)[0]
    match name.lower():
        case "euclidean":
            cost = x1[:, None] - x2[None, :]
            cost = keras.ops.reshape(cost, (n, m, -1))
            cost = keras.ops.norm(cost, ord=2, axis=-1)
        case other:
            raise ValueError(f"Unsupported cost name: '{other}'.")

    return cost


@find_cost.register(BackendTensor)
def _(cost: Tensor, *args, **kwargs):
    return cost
