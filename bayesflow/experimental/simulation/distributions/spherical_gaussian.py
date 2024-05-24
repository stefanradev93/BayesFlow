
import math

import keras
from keras import ops

from bayesflow.experimental.types import Shape, Distribution, Tensor


class SphericalGaussian(Distribution):
    """Utility class for a backend-agnostic spherical Gaussian distribution.

    Note:
        - ``log_unnormalized_pdf`` method is used as a loss function
        - ``log_pdf`` is used for density computation
    """
    def __init__(self, shape: Shape):
        self.shape = shape
        self.dim = int(self.shape[0])
        self._norm_const = 0.5 * self.dim * math.log(2.0 * math.pi)

    def sample(self, batch_shape: Shape):
        return keras.random.normal(shape=batch_shape + self.shape, mean=0.0, stddev=1.0)

    def log_unnormalized_prob(self, tensor: Tensor):
        return -0.5 * ops.sum(ops.square(tensor), axis=-1)

    def log_prob(self, tensor: Tensor):
        log_unnorm_pdf = self.log_unnormalized_prob(tensor)
        return log_unnorm_pdf - self._norm_const
