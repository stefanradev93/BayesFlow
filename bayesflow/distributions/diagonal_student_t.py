import keras
from keras.saving import register_keras_serializable as serializable

import math
import numpy as np

from scipy.stats import t as scipy_student_t

from bayesflow.types import Shape, Tensor
from .distribution import Distribution


@serializable(package="bayesflow.distributions")
class DiagonalStudentT(Distribution):
    """Implements a backend-agnostic diagonal Student-t distribution."""

    def __init__(
        self,
        df: int | float,
        loc: int | float | np.ndarray | Tensor = 0.0,
        scale: int | float | np.ndarray | Tensor = 1.0,
        use_learnable_parameters: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.df = df
        self.loc = loc
        self.scale = scale

        self.dim = None
        self.log_normalization_constant = None

        self.use_learnable_parameters = use_learnable_parameters

    def build(self, input_shape: Shape) -> None:
        self.dim = int(input_shape[-1])

        # convert to tensor and broadcast if necessary
        self.loc = keras.ops.broadcast_to(self.loc, (self.dim,))
        self.loc = keras.ops.cast(self.loc, "float32")

        self.scale = keras.ops.broadcast_to(self.scale, (self.dim,))
        self.scale = keras.ops.cast(self.scale, "float32")

        self.log_normalization_constant = (
            -0.5 * self.dim * math.log(self.df)
            - 0.5 * self.dim * math.log(math.pi)
            - math.lgamma(0.5 * self.df)
            + math.lgamma(0.5 * (self.df + self.dim))
            - keras.ops.sum(keras.ops.log(self.scale))
        )

        if self.use_learnable_parameters:
            loc = self.loc
            self.loc = self.add_weight(
                shape=keras.ops.shape(loc),
                initializer="zeros",
                dtype="float32",
            )
            self.loc.assign(loc)

            scale = self.scale
            self.scale = self.add_weight(
                shape=keras.ops.shape(scale),
                initializer="ones",
                dtype="float32",
            )
            self.scale.assign(scale)

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        mahalanobis_term = keras.ops.sum((samples - self.loc) ** 2 / self.scale**2, axis=-1)
        result = -0.5 * (self.df + self.dim) * keras.ops.log1p(mahalanobis_term / self.df)

        if normalize:
            result += self.log_normalization_constant

        return result

    def sample(self, batch_shape: Shape) -> Tensor:
        # TODO: use reparameterization trick instead of scipy
        # TODO: use the seed generator state
        dist = scipy_student_t(df=self.df, loc=self.loc, scale=self.scale)
        samples = dist.rvs(size=batch_shape + (self.dim,))

        return keras.ops.convert_to_tensor(samples)
