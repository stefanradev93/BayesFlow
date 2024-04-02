
import keras

from .permutation import Permutation


class Shuffle(Permutation):
    def build(self, input_shape):
        forward_indices = keras.ops.arange(input_shape[0])
        forward_indices = keras.random.shuffle(forward_indices)
        inverse_indices = keras.ops.argsort(forward_indices)

        self.forward_indices = keras.Variable(forward_indices, name="forward_indices", trainable=False)
        self.inverse_indices = keras.Variable(inverse_indices, name="inverse_indices", trainable=False)
