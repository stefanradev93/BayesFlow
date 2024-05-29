
import keras
from keras.saving import register_keras_serializable

from .distribution import Distribution


@register_keras_serializable(package="bayesflow.distributions")
class Normal(Distribution):
    def __init__(self, mean: float = 0.0, std: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.shape = None

    def build(self, input_shape):
        self.shape = list(input_shape[-1:])

    def sample(self, batch_shape):
        return keras.random.normal(batch_shape + self.shape, mean=self.mean, stddev=self.std)
