import keras
from typing import TypeVar

match keras.backend.backend():
    case "numpy":
        import numpy as np

        bound = np.ndarray
    case "jax":
        import jax

        bound = jax.Array
    case "tensorflow":
        import tensorflow as tf

        bound = tf.Tensor
    case "torch":
        import torch

        bound = torch.Tensor
    case other:
        raise NotImplementedError

Tensor = TypeVar("Tensor", bound=bound)
