
from typing import NotRequired, TypedDict

Shape = tuple[int, ...]

# this is ugly, but:
# 1. it is recognized by static type checkers (not possible with if-else branching)
# 2. it does not leave the Tensor type possibly undefined (not possible without nesting)
try:
    import jax
    Tensor = jax.Array
except ModuleNotFoundError:
    try:
        import tensorflow as tf
        Tensor = tf.Tensor
    except ModuleNotFoundError:
        import torch
        Tensor = torch.Tensor

Context = Tensor
Observable = Tensor
Parameter = Tensor
Summary = Tensor

Contexts = dict[str, Context]
Observables = dict[str, Observable]
Parameters = dict[str, Parameter]
Summaries = dict[str, Summary]


data_keys = frozenset(["contexts", "observables", "parameters", "summaries"])


Data = dict[str, Contexts | Observables | Parameters | Summaries]

class DataDict(TypedDict):
    contexts: NotRequired[Contexts]
    observables: NotRequired[Observables]
    parameters: NotRequired[Parameters]
