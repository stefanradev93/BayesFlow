import keras

from bayesflow.types import Tensor


def repeat_tensor(tensor: Tensor, num_repeats: int, axis=1):
    """Utility function to repeat a tensor over a given axis ``num_repeats`` times."""

    tensor = keras.ops.expand_dims(tensor, axis=axis)
    repeats = [1] * tensor.ndim
    repeats[axis] = num_repeats
    repeated_tensor = keras.ops.tile(tensor, repeats=repeats)
    return repeated_tensor
