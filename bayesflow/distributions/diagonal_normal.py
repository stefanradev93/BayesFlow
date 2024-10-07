import math

import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Shape, Tensor
from .distribution import Distribution


@serializable(package="bayesflow.distributions")
class DiagonalNormal(Distribution):
    """Implements a backend-agnostic spherical Gaussian distribution."""

    def __init__(
        self,
        mean: float | Tensor = 0.0,
        std: float | Tensor = 1.0,
        seed_generator: keras.random.SeedGenerator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.var = std**2
        self.dim = None
        self.log_normalization_constant = None
        self.seed_generator = seed_generator or keras.random.SeedGenerator()

    def build(self, input_shape: Shape) -> None:
        self.dim = int(input_shape[-1])
        self.log_normalization_constant = 0.5 * self.dim * (math.log(2.0 * math.pi) + math.log(self.var))

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        result = -0.5 * keras.ops.sum((samples - self.mean) ** 2, axis=-1) / self.var

        if normalize:
            result -= self.log_normalization_constant

        return result

    def sample(self, batch_shape: Shape) -> Tensor:
        return keras.random.normal(
            shape=batch_shape + (self.dim,), mean=self.mean, stddev=self.std, seed=self.seed_generator
        )
