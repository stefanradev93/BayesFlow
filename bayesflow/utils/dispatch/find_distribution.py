
from functools import singledispatch


@singledispatch
def find_distribution(arg, **kwargs):
    raise TypeError(f"Cannot infer distribution from {arg!r}.")


@find_distribution.register
def _(name: str, **kwargs):
    match name.lower():
        case "normal":
            from bayesflow.distributions import DiagonalNormal
            distribution = DiagonalNormal(**kwargs)
        case other:
            raise ValueError(f"Unsupported distribution name '{other}'.")

    return distribution


@find_distribution.register
def _(constructor: type, **kwargs):
    return constructor(**kwargs)
