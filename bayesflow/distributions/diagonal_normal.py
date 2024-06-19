
import math

import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from .distribution import Distribution


@keras.saving.register_keras_serializable(package="bayesflow.distributions")
class DiagonalNormal(Distribution):
    """Utility class for a backend-agnostic spherical Gaussian distribution.

    Note:
        - ``_log_unnormalized_prob`` method is used as a loss function
        - ``log_prob`` is used for density computation
    """
    def __init__(self, mean: float | Tensor = 0.0, std: float | Tensor = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.var = std**2
        self.dim = None
        self.log_norm_const = None

    def sample(self, batch_shape: Shape) -> Tensor:
        return keras.random.normal(shape=batch_shape + (self.dim,), mean=self.mean, stddev=self.std)

    def log_prob(self, tensor: Tensor) -> Tensor:
        log_unnorm_pdf = self._log_unnormalized_prob(tensor)
        return log_unnorm_pdf - self.log_norm_const

    def build(self, input_shape: Shape) -> None:
        self.dim = int(input_shape[-1])
        self.log_norm_const = 0.5 * self.dim * (math.log(2.0 * math.pi) + math.log(self.var))

    def _log_unnormalized_prob(self, tensor: Tensor) -> Tensor:
        return -0.5 * ops.sum(ops.square(tensor - self.mean), axis=-1) / self.var
