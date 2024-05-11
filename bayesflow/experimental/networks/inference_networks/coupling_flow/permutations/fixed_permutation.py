
import keras
from keras import ops

from bayesflow.experimental.types import Tensor


class FixedPermutation(keras.layers.Layer):
    def __init__(self, indices: Tensor):
        super().__init__()
        self.indices = indices
        self.inverse_indices = ops.argsort(indices, axis=0)

    def forward(self, x: Tensor) -> Tensor:
        return ops.take(x, self.indices, axis=-1)

    def inverse(self, z: Tensor) -> Tensor:
        return ops.take(z, self.inverse_indices, axis=-1)

    @classmethod
    def swap(cls, target_dim: int):
        indices = ops.arange(0, target_dim)
        indices = ops.roll(indices, shift=target_dim // 2, axis=0)
        return cls(indices)

    @classmethod
    def random(cls, target_dim: int):
        indices = ops.arange(0, target_dim)
        indices = keras.random.shuffle(indices)
        return cls(indices)
