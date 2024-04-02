
import keras

from bayesflow.experimental.types import Tensor


class Permutation(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.forward_indices = None
        self.inverse_indices = None

    def build(self, input_shape):
        """ Initialize the indices as a non-trainable keras.Variable """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return x[:, self.forward_indices]

    def inverse(self, z: Tensor) -> Tensor:
        return z[:, self.inverse_indices]
