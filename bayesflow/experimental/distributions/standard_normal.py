
import keras
from keras.saving import register_keras_serializable

import numpy as np

from .distribution import Distribution


@register_keras_serializable(package="bayesflow.distributions")
class StandardNormal(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = None

    def build(self, input_shape):
        self.shape = input_shape[-1:]

    def sample(self, batch_shape):
        return keras.random.normal(batch_shape + self.shape)

    def log_prob(self, samples):
        return -0.5 * keras.ops.sum(keras.ops.square(samples), axis=-1) - 0.5 * self.shape[0] * keras.ops.log(2.0 * np.pi)
