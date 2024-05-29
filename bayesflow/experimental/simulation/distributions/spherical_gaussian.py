
import math

import keras
from keras import ops

from bayesflow.experimental.types import Shape, Distribution, Tensor


@keras.saving.register_keras_serializable(package="bayesflow.simulation")
class SphericalGaussian(Distribution):
    """Utility class for a backend-agnostic spherical Gaussian distribution.

    Note:
        - ``log_unnormalized_prob`` method is used as a loss function
        - ``log_prob`` is used for density computation
    """
    def __init__(self):
        self.dim = None
        self.log_norm_const = None

    def sample(self, batch_shape: Shape):
        return keras.random.normal(shape=batch_shape + (self.dim,), mean=0.0, stddev=1.0)

    def log_unnormalized_prob(self, tensor: Tensor):
        return -0.5 * ops.sum(ops.square(tensor), axis=-1)

    def log_prob(self, tensor: Tensor):
        log_unnorm_pdf = self.log_unnormalized_prob(tensor)
        return log_unnorm_pdf - self.log_norm_const

    def build(self, input_shape):
        self.dim = int(input_shape[-1])
        self.log_norm_const = 0.5 * self.dim * math.log(2.0 * math.pi)
