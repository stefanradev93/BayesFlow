
import warnings

import keras
import tensorflow as tf


class TensorFlowGenerativeTrainer(keras.Model):
    def __init__(self, model: keras.Model):
        super().__init__()
        self.model = model

    def compute_loss_metrics(self, data):
        if hasattr(self.model, "compute_loss_metrics"):
            return self.model.compute_loss_metrics(data)

        # default implementation, may be inefficient as it computes the loss and metrics separately
        return self.model.compute_loss(data), self.model.compute_metrics(data, None, None)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, metrics = self.compute_loss_metrics(data)
            self._loss_tracker.update_state(loss)
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            trainable_weights = self.trainable_weights
            gradients = tape.gradient(loss, trainable_weights)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        else:
            warnings.warn("The model does not have any trainable weights.")

        return metrics

    def test_step(self, data):
        loss, metrics = self.compute_loss_metrics(data)
        self._loss_tracker.update_state(loss)
        return metrics

    def predict_step(self, data):
        raise RuntimeError(f"Generative Models do not support a predict step. Perhaps you want to sample instead?")
