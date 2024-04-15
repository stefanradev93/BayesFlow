
import keras


class Permutation(keras.Layer):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices
        self.inverse_indices = keras.ops.argsort(indices, axis=0)

    def forward(self, x):
        return x[:, self.indices]

    def inverse(self, x):
        return x[:, self.inverse_indices]

    @classmethod
    def identity(cls, num_features):
        return cls(keras.ops.arange(num_features))

    @classmethod
    def random(cls, num_features):
        indices = keras.random.shuffle(keras.ops.arange(num_features))
        return cls(indices)

    @classmethod
    def reverse(cls, num_features):
        return cls(keras.ops.arange(num_features)[::-1])

    @classmethod
    def swap(cls, num_features):
        indices = keras.ops.arange(num_features)
        indices = keras.ops.roll(indices, num_features // 2)
        return cls(indices)
