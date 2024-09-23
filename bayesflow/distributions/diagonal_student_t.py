import math

from scipy.stats import t as scipy_student_t

import keras
from keras.saving import register_keras_serializable as serializable
from keras import ops

from bayesflow.types import Shape, Tensor
from .distribution import Distribution


@serializable(package="bayesflow.distributions")
class DiagonalStudentT(Distribution):
    """Utility class for a backend-agnostic spherical Student-t distribution.

    Note:
        - ``_log_unnormalized_prob`` method is used as a loss function
        - ``log_prob`` is used for density computation
    """

    def __init__(
        self,
        df: float = 50,
        loc: float | Tensor = 0.0,
        scale: float | Tensor = 1.0,
        seed_generator: keras.random.SeedGenerator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loc = loc
        self.scale = scale
        self.df = df
        self.dim = None
        self.log_norm_const = None
        self.seed_generator = seed_generator or keras.random.SeedGenerator()

    def sample(self, batch_shape: Shape) -> Tensor:
        dist = scipy_student_t(df=self.df, loc=self.loc, scale=self.scale)
        return dist.rvs(size=batch_shape)

    def log_prob(self, tensor: Tensor) -> Tensor:
        log_unnorm_pdf = self._log_unnormalized_prob(tensor)
        return log_unnorm_pdf - self.log_norm_const

    def build(self, input_shape: Shape) -> None:
        self.dim = int(input_shape[-1])
        self.log_norm_const = self.dim * (
            math.lgamma(self.df / 2.0)
            - math.lgamma((self.df + 1.0) / 2.0)
            + 0.5 * math.log(self.df * math.pi)
            + math.log(self.scale)
        )

    def _log_unnormalized_prob(self, tensor: Tensor) -> Tensor:
        return (
            -(self.df + 1.0)
            / 2.0
            * ops.sum(ops.log(1.0 + 1.0 / self.df * ops.square((tensor - self.loc) / self.scale)), axis=-1)
        )
