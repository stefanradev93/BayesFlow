
import keras
from functools import singledispatch


@singledispatch
def find_permutation(arg, **kwargs):
    raise TypeError(f"Cannot infer permutation from {arg!r}.")


@find_permutation.register
def _(name: str, **kwargs):
    match name.lower():
        case "random":
            from bayesflow.networks.coupling_flow.permutations import RandomPermutation
            return RandomPermutation(**kwargs)
        case "swap":
            from bayesflow.networks.coupling_flow.permutations import Swap
            return Swap(**kwargs)
        case "learnable" | "orthogonal":
            from bayesflow.networks.coupling_flow.permutations import OrthogonalPermutation
            return OrthogonalPermutation(**kwargs)


@find_permutation.register
def _(permutation: keras.Layer, **kwargs):
    return permutation


@find_permutation.register
def _(none: None, **kwargs):
    return None
