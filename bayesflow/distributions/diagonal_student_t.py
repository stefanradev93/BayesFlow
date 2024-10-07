import math

from scipy.stats import t as scipy_student_t

import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Shape, Tensor
from .distribution import Distribution


@serializable(package="bayesflow.distributions")
class DiagonalStudentT(Distribution):
    """Implements a backend-agnostic spherical Student-t distribution."""

    def __init__(
        self,
        degrees_of_freedom: int = 50,
        mean: float | Tensor = 0.0,
        std: float | Tensor = 1.0,
        seed_generator: keras.random.SeedGenerator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.degrees_of_freedom = degrees_of_freedom
        self.mean = mean
        self.std = std
        self.seed_generator = seed_generator or keras.random.SeedGenerator()

        self.dim = None
        self.log_normalization_constant = None

    def build(self, input_shape: Shape) -> None:
        self.dim = int(input_shape[-1])
        self.log_normalization_constant = (
            0.5 * self.dim * math.log(self.degrees_of_freedom)
            + 0.5 * self.dim * math.log(math.pi)
            + math.lgamma(0.5 * self.degrees_of_freedom)
            - math.lgamma(0.5 * (self.degrees_of_freedom + self.dim))
            + 0.5 * keras.ops.sum(keras.ops.log(self.std))
        )

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        mahalanobis_term = keras.ops.sum((samples - self.mean) ** 2 / self.std**2, axis=-1)
        result = (
            -0.5 * (self.degrees_of_freedom + self.dim) * keras.ops.log1p(mahalanobis_term / self.degrees_of_freedom)
        )

        if normalize:
            result -= self.log_normalization_constant

        return result

    def sample(self, batch_shape: Shape) -> Tensor:
        # TODO: use reparameterization trick instead of scipy
        # TODO: use the seed generator state
        dist = scipy_student_t(df=self.degrees_of_freedom, loc=self.mean, scale=self.std)
        samples = dist.rvs(size=batch_shape + (self.dim,))

        return keras.ops.convert_to_tensor(samples)
