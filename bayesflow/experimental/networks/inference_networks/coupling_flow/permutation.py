
import keras

from bayesflow.experimental.types import Tensor


class Permutation(keras.layers.Layer):
    def __init__(self, indices: Tensor):
        super().__init__()
        self.indices = indices
        self.inverse_indices = keras.ops.argsort(indices, axis=0)

    def forward(self, x: Tensor) -> Tensor:
        return keras.ops.take(x, self.indices, axis=1)

    def inverse(self, z: Tensor) -> Tensor:
        return keras.ops.take(z, self.inverse_indices, axis=1)

    @classmethod
    def swap(cls, features: int):
        indices = keras.ops.arange(0, features)
        indices = keras.ops.roll(indices, shift=features // 2, axis=0)
        return cls(indices)

    @classmethod
    def random(cls, features: int):
        indices = keras.ops.arange(0, features)
        indices = keras.random.shuffle(indices)
        return cls(indices)
