from collections.abc import Callable, Mapping, Sequence
from functools import singledispatch
import numpy as np
from types import FunctionType

from .simulator import Simulator


@singledispatch
def make_simulator(arg, *_, **__):
    raise TypeError(f"Cannot infer simulator from {arg!r}.")


@make_simulator.register
def _(simulator: Simulator):
    return simulator


@make_simulator.register
def _(fn: FunctionType, **kwargs):
    from bayesflow.simulators import LambdaSimulator

    return LambdaSimulator(fn, **kwargs)


@make_simulator.register
def _(
    objs: Sequence,
    obj_kwargs: Mapping[str, dict[str, any]] = None,
    meta_fn: Callable[[], dict[str, np.ndarray]] = None,
    **kwargs,
):
    from bayesflow.simulators import LambdaSimulator, SequentialSimulator

    if obj_kwargs is None:
        obj_kwargs = {}

    simulators = []

    for obj in objs:
        if hasattr(obj, "__name__"):
            obj_kwargs = obj_kwargs.get(obj.__name__, {})
        else:
            obj_kwargs = {}

        simulators.append(make_simulator(obj, **obj_kwargs))

    if meta_fn is not None:
        meta = LambdaSimulator(meta_fn, is_batched=True)
        simulators = [meta, *simulators]

    return SequentialSimulator(simulators, **kwargs)


@make_simulator.register
def _(
    objs: Mapping,
    obj_kwargs: Mapping[str, dict[str, any]] = None,
    meta_fn: Callable[[], dict[str, np.ndarray]] = None,
    **kwargs,
):
    from bayesflow.simulators import LambdaSimulator, SequentialSimulator

    if obj_kwargs is None:
        obj_kwargs = {}

    simulators = []

    for name, obj in objs.items():
        obj_kwargs = obj_kwargs.get(name, {})
        simulators.append(make_simulator(obj, **obj_kwargs))

    if meta_fn is not None:
        meta = LambdaSimulator(meta_fn, is_batched=True)
        simulators = [meta, *simulators]

    return SequentialSimulator(simulators, **kwargs)
