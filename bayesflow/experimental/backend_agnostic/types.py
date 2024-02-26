Shape = int | tuple[int, ...]

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


Contexts = dict[str, Tensor]
Observations = dict[str, Tensor]
Parameters = dict[str, Tensor]
