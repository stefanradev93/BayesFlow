import keras
import torch

from bayesflow.utils import filter_kwargs


class TorchApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        # implemented by each respective architecture
        raise NotImplementedError

    def test_step(self, data: dict[str, any]) -> dict[str, torch.Tensor]:
        kwargs = filter_kwargs(data | {"stage": "validation"}, self.compute_metrics)
        return self.compute_metrics(**kwargs)

    def train_step(self, data: dict[str, any]) -> dict[str, torch.Tensor]:
        with torch.enable_grad():
            kwargs = filter_kwargs(data | {"stage": "training"}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

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
