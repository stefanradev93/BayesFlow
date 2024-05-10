
import keras
from keras import ops as K

from bayesflow.experimental.types import Tensor


class FixedPermutation(keras.layers.Layer):
    def __init__(self, indices: Tensor):
        super().__init__()
        self.indices = indices
        self.inverse_indices = K.argsort(indices, axis=0)

    def forward(self, x: Tensor) -> Tensor:
        return K.take(x, self.indices, axis=-1)

    def inverse(self, z: Tensor) -> Tensor:
        return K.take(z, self.inverse_indices, axis=-1)

    @classmethod
    def swap(cls, target_dim: int):
        indices = K.arange(0, target_dim)
        indices = K.roll(indices, shift=target_dim // 2, axis=0)
        return cls(indices)

    @classmethod
    def random(cls, target_dim: int):
        indices = K.arange(0, target_dim)
        indices = keras.random.shuffle(indices)
        return cls(indices)
