import math

from scipy.stats import t as scipy_student_t

import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from .distribution import Distribution


@keras.saving.register_keras_serializable(package="bayesflow.distributions")
class DiagonalStudentT(Distribution):
    """Utility class for a backend-agnostic spherical Gaussian distribution.

    Note:
        - ``_log_unnormalized_prob`` method is used as a loss function
        - ``log_prob`` is used for density computation
    """

    def __init__(self, df: float = 50, loc: float | Tensor = 0.0, scale: float | Tensor = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.loc = loc
        self.scale = scale
        self.df = df
        self.dim = None
        self.log_norm_const = None

    def sample(self, batch_shape: Shape) -> Tensor:
        dist = scipy_student_t(df=self.df, loc=self.loc, scale=self.scale)
        return dist.rvs(size=batch_shape)

    def log_prob(self, tensor: Tensor) -> Tensor:
        #TODO
        pass

    def build(self, input_shape: Shape) -> None:
        #TODO
        pass

    def _log_unnormalized_prob(self, tensor: Tensor) -> Tensor:
        #TODO
        pass
