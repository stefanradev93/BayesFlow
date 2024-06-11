
import jax
import keras

from .base_approximator import BaseApproximator
from ..types import Tensor


class JAXApproximator(BaseApproximator):
    # noinspection PyMethodOverriding
    def train_step(self, state: any, data: dict[str, Tensor]) -> (any, dict[str, Tensor]):
        # TODO: not functional yet
        grad_fn = jax.value_and_grad(self.compute_metrics)

        # TODO: account for updating non-trainable variables (batchnorm etc.)
        metrics, grads = grad_fn(state, data)

        state = self.update_state(state, grads)

        # keras turns these around for some reason
        return metrics, state

    def update_state(self, state: any, grads: any) -> any:
        trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables = state

        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(grads, trainable_variables)

        state = trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables

        return state

    # noinspection PyMethodOverriding
    def compute_metrics(self, state: any, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        raise NotImplementedError
