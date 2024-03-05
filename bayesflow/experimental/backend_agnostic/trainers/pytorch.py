import warnings

import keras
import torch


class PyTorchGenerativeTrainer(keras.Model):
    def __init__(self, model: keras.Model):
        super().__init__()
        self.model = model

    def compute_loss_metrics(self, data):
        if hasattr(self.model, "compute_loss_metrics"):
            return self.model.compute_loss_metrics(data)

        # default implementation, may be inefficient as it computes the loss and metrics separately
        return self.model.compute_loss(data), self.model.compute_metrics(data, None, None)

    def train_step(self, data):
        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        loss, metrics = self.compute_loss_metrics(data)
        self._loss_tracker.update_state(loss)
        if self.optimizer is not None:
            loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            # Call torch.Tensor.backward() on the loss to compute gradients
            # for the weights.
            loss.backward()

            trainable_weights = self.trainable_weights[:]
            gradients = [v.value.grad for v in trainable_weights]

            # Update weights
            with torch.no_grad():
                self.optimizer.apply(gradients, trainable_weights)
        else:
            warnings.warn("The model does not have any trainable weights.")

        return metrics

    def test_step(self, data):
        loss, metrics = self.compute_loss_metrics(data)
        self._loss_tracker.update_state(loss)
        return metrics

    def predict_step(self, data):
        raise RuntimeError(f"Generative Models do not support a predict step. Perhaps you want to sample instead?")
