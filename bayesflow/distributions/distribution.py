
import keras

from bayesflow.types import Shape, Tensor


class Distribution(keras.Layer):
    def call(self, samples: Tensor) -> Tensor:
        return keras.ops.exp(self.log_prob(samples))

    def sample(self, batch_shape: Shape) -> Tensor:
        raise NotImplementedError

    def log_prob(self, samples: Tensor) -> Tensor:
        raise NotImplementedError
