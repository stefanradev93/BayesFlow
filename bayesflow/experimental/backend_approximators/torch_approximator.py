
import torch

from .base_approximator import BaseApproximator


class TorchApproximator(BaseApproximator):
    def train_step(self, data):
        with torch.enable_grad():
            metrics = self.compute_metrics(data, stage="training")

        loss = metrics["loss"]

        # noinspection PyUnresolvedReferences
        self.zero_grad()
        loss.backward()

        trainable_weights = self.trainable_weights[:]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self._loss_tracker.update_state(loss)

        return metrics
