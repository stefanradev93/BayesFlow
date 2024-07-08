from typing import TYPE_CHECKING

Shape = tuple[int, ...]


if TYPE_CHECKING:
    import keras

    match keras.backend.backend():
        case "numpy":
            import numpy as np

            Tensor = np.ndarray
        case "jax":
            import jax

            Tensor = jax.Array
        case "tensorflow":
            import tensorflow as tf

            Tensor = tf.Tensor
        case "torch":
            import torch

            Tensor = torch.Tensor
        case other:
            raise NotImplementedError

else:
    import keras

    Tensor = type(keras.ops.zeros(()))
