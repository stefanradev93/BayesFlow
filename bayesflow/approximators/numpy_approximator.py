from bayesflow.types import Tensor

from .base_approximator import BaseApproximator


class NumpyApproximator(BaseApproximator):
    def train_step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError("Keras currently has no support for numpy training.")
