
import tensorflow as tf

from bayesflow.experimental.types import Tensor

from .base_approximator import BaseApproximator


class TensorFlowApproximator(BaseApproximator):
    def train_step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        with tf.GradientTape() as tape:
            metrics = self.compute_metrics(data)

        loss = metrics["loss"]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._loss_tracker.update_state(loss)

        return metrics
