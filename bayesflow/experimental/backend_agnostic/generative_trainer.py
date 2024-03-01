
import warnings

import keras


class GenerativeTrainer(keras.Model):
    """
    Implements a custom training scheme for generative modeling with support for multiple backends.
    """
    def __init__(self, model: keras.Model):
        super().__init__()
        self.model = model

    def train_step(self, *args, **kwargs):
        backend = keras.backend.backend()

        try:
            return getattr(self, f"train_step_{backend}")(*args, **kwargs)
        except AttributeError:
            raise NotImplementedError(f"Generative model training is not supported for backend '{backend}'.")

    def compute_loss_metrics(self, data):
        if hasattr(self.model, "compute_loss_metrics"):
            return self.model.compute_loss_metrics(data)

        # default implementation, may be inefficient as it computes the loss and metrics separately
        return self.model.compute_loss(data), self.model.compute_metrics(data, None, None)

    def train_step_tensorflow(self, data):
        import tensorflow as tf

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

    def train_step_torch(self, data):
        import torch

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

    def train_step_jax(self, state, data):
        # TODO: jax is weird
        raise NotImplementedError("JAX support is not yet implemented.")

    def train_step_numpy(self, data):
        # TODO: apparently this is also a thing in keras
        raise NotImplementedError("Numpy support is not yet implemented.")
