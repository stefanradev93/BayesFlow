
import numpy as np

from bayesflow.experimental.types import Tensor

from .base_approximator import BaseApproximator


class NumpyApproximator(BaseApproximator):
    def train_step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError(f"Keras currently has no support for numpy training.")
