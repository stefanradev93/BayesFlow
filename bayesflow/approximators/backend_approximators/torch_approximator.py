import keras
import torch


class TorchApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, data: any, stage: str = "training") -> dict[str, torch.Tensor]:
        # implemented by each respective architecture
        raise NotImplementedError

    def train_step(self, data: any) -> dict[str, torch.Tensor]:
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
