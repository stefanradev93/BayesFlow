import math

import numpy as np
from scipy.stats import t as scipy_student_t

import keras
from keras.saving import register_keras_serializable as serializable
from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils import expand_left_as
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
        loc: float | np.ndarray = 0.0,
        scale: float | np.ndarray = 1.0,
        dynamic_params: bool = False,
        seed_generator: keras.random.SeedGenerator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Handle scale and location array vs scalar arguments
        if isinstance(loc, np.ndarray) and isinstance(scale, np.ndarray):
            if loc.shape[0] != scale.shape[0]:
                raise ValueError(
                    f"Shapes of loc and scale must be equal, " f"but are currently {loc.shape} and {scale.shape}"
                )
        elif isinstance(loc, np.ndarray) and isinstance(scale, (int, float)):
            scale = np.full(loc.shape[0], scale)
        elif isinstance(loc, (int, float)) and isinstance(scale, np.ndarray):
            loc = np.full(scale.shape[0], loc)

        # Set dim based on whether loc is an array, or keep it None for scalar inputs
        self.dim = loc.shape[0] if isinstance(loc, np.ndarray) else None

        self.loc = loc
        self.scale_diag = scale
        self.df = df
        self.log_norm_const = None
        self.dynamic_params = dynamic_params
        self.seed_generator = seed_generator or keras.random.SeedGenerator()

    def sample(self, batch_shape: Shape) -> Tensor:
        dist = scipy_student_t(df=self.df, loc=self.loc, scale=self.scale_diag)
        return dist.rvs(size=batch_shape)

    def log_prob(self, tensor: Tensor) -> Tensor:
        log_unnorm_pdf = self._log_unnormalized_prob(tensor)
        return log_unnorm_pdf - self.log_norm_const

    def build(self, input_shape: Shape) -> None:
        dim = int(input_shape[-1])

        # At this point, loc and scale are either vectors or scalars, so it suffices to check any of them
        if isinstance(self.loc, np.ndarray):
            if self.loc.shape[0] != dim:
                raise ValueError(
                    f"The dimensions of your inference variables should be {self.loc.shape[0]} "
                    f"but are currently {dim}. Please, ensure that dimensions match."
                )
        else:
            self.loc = np.full(dim, self.loc)
            self.scale_diag = np.full(dim, self.scale_diag)

        self.dim = dim

        # Pre-compute log normalizing constant
        log_norm_const = -0.5 * self.dim * math.log(self.df * math.pi)
        log_norm_const += math.lgamma(0.5 * (self.df + self.dim))
        log_norm_const -= math.lgamma(0.5 * self.df)
        log_norm_const -= np.sum(np.log(self.scale_diag))

        # Add distribution parameters as keras weights
        self.log_norm_const = self.add_weight(
            name="log_norm_const",
            shape=(1,),
            trainable=False,
            dtype="float32",
            initializer=keras.initializers.Constant(log_norm_const),
        )
        self.df = self.add_weight(
            name="dof", shape=(1,), trainable=False, dtype="float32", initializer=keras.initializers.Constant(self.df)
        )
        self.loc = self.add_weight(
            name="loc",
            shape=(self.dim,),
            trainable=self.dynamic_params,
            dtype="float32",
            initializer=keras.initializers.Constant(self.loc),
        )
        self.scale_diag = self.add_weight(
            name="scale_diag",
            shape=(self.dim,),
            trainable=self.dynamic_params,
            dtype="float32",
            initializer=keras.initializers.Constant(self.scale_diag),
        )

    def _log_unnormalized_prob(self, tensor: Tensor) -> Tensor:
        log_unnormalized_pdf = -0.5 * (self.df + self.dim)
        log_unnormalized_pdf *= ops.log(
            1
            + (1 / self.df)
            * ops.sum(
                (tensor - expand_left_as(self.loc, tensor)) ** 2 / expand_left_as(self.scale_diag, tensor) ** 2, axis=-1
            )
        )
        return log_unnormalized_pdf
