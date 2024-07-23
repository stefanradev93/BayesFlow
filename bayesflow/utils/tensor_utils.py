import logging

import keras

from bayesflow.types import Tensor


def repeat_tensor(tensor: Tensor, num_repeats: int, axis=1) -> Tensor:
    """Utility function to repeat a tensor over a given axis ``num_repeats`` times."""

    tensor = keras.ops.expand_dims(tensor, axis=axis)
    repeats = [1] * tensor.ndim
    repeats[axis] = num_repeats
    repeated_tensor = keras.ops.tile(tensor, repeats=repeats)
    return repeated_tensor


def process_output(outputs: Tensor, convert_to_numpy: bool = True) -> Tensor:
    """Utility function to apply common post-processing steps to the outputs of an approximator."""

    # Remove trailing first axis for single data sets
    if keras.ops.shape(outputs)[0] == 1:
        outputs = keras.ops.squeeze(outputs, axis=0)

    # Warn if any NaNs present in output
    nan_mask = keras.ops.isnan(outputs)
    if keras.ops.any(nan_mask):
        logging.warning(f"A total of {keras.ops.sum(nan_mask)} NaNs found in output.")

    # Warn if any inf present in output
    inf_mask = keras.ops.isinf(outputs)
    if keras.ops.any(inf_mask):
        logging.warning(f"A total of {keras.ops.sum(inf_mask)} inf values found in output.")

    if convert_to_numpy:
        return outputs.numpy()
    return outputs
