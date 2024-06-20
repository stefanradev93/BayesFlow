from collections.abc import Callable


Shape = tuple[int, ...]

# this is ugly, but:
# 1. it is recognized by static type checkers (not possible with if-else branching)
# 2. it does not leave the Tensor type possibly undefined (not possible without nesting)
try:
    import jax

    Tensor: type(jax.Array) = jax.Array
except ModuleNotFoundError:
    try:
        import tensorflow as tf

        Tensor: type(tf.Tensor) = tf.Tensor
    except ModuleNotFoundError:
        import torch

        Tensor: type(torch.Tensor) = torch.Tensor


BatchedConditionalSampler = Callable[[Shape, Tensor, ...], dict[str, Tensor]]
BatchedUnconditionalSampler = Callable[[Shape], dict[str, Tensor]]
UnbatchedConditionalSampler = Callable[[Tensor, ...], dict[str, Tensor]]
UnbatchedUnconditionalSampler = Callable[[], dict[str, Tensor]]

Sampler = (
    BatchedConditionalSampler
    | UnbatchedConditionalSampler
    | UnbatchedConditionalSampler
    | UnbatchedUnconditionalSampler
)
