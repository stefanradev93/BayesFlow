
import keras

from .permutation import Permutation


class Swap(Permutation):
    def build(self, input_shape):
        indices = keras.ops.arange(input_shape[0])
        shift = input_shape[0] // 2
        forward_indices = keras.ops.roll(indices, +shift)
        inverse_indices = keras.ops.roll(indices, -shift)

        self.forward_indices = keras.Variable(forward_indices, name="forward_indices", trainable=False)
        self.inverse_indices = keras.Variable(inverse_indices, name="inverse_indices", trainable=False)
