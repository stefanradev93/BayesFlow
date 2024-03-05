
import keras
from keras.src import backend

import jax


class JAXGenerativeTrainer(keras.Model):
    def __init__(self, model: keras.Model):
        super().__init__()
        self.model = model

    def compute_loss_metrics_stateless(self, trainable_variables, non_trainable_variables, data, training=False, optimizer_variables=None):
        if hasattr(self.model, "compute_loss_metrics_stateless"):
            return self.model.compute_loss_metrics_stateless(trainable_variables, non_trainable_variables, data, training, optimizer_variables)

        state_mapping = list(zip(self.trainable_variables, trainable_variables))
        state_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables))

        with backend.StatelessScope(state_mapping) as scope:
            if hasattr(self.model, "compute_loss_metrics"):
                loss, metrics = self.model.compute_loss_metrics(data)
            else:
                loss = self.model.compute_loss(data)
                metrics = self.model.compute_metrics(data, None, None)

            self._loss_tracker.update_state(loss)

        if training and self.optimizer is not None:
            # Scale loss with a StatelessScope, to use an update scale variable.
            mapping = list(zip(self.optimizer.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping):
                loss = self.optimizer.scale_loss(loss)



        return loss, metrics, state

    @jax.jit
    def train_step(self, state, data):
        super().train_step()
        trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables = state

        grad_fn = jax.value_and_grad(self.compute_loss_metrics_stateless, has_aux=True)

        ((loss, metrics), (non_trainable_variables, metrics_variables)), grads = grad_fn(trainable_variables, non_trainable_variables, data, training=True, optimizer_variables=optimizer_variables)

        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        state = self._enforce_jax_state_sharding(
            trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables
        )

        return metrics, state

    def compute_loss_metrics_stateless(self, trainable_variables, non_trainable_variables, metrics_variables, data, training=False, optimizer_variables=None):
        loss = ...

        return loss, (non_trainable_variables, metrics_variables)

    @jax.jit
    def train_step(self, state, data):
        trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables = state

        grad_fn = jax.value_and_grad(self.compute_loss_metrics_stateless, has_aux=True)

        (loss, (non_trainable_variables, metrics_variables)), grads = grad_fn(
            trainable_variables, non_trainable_variables, metrics_variables, data, training=True, optimizer_variables=optimizer_variables
        )

        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        state = trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables

        return loss, state

    def eval_step

    def test_step(self, state, data):
        # TODO
        ...

    def predict_step(self, state, data):
        raise RuntimeError(f"Generative Models do not support a predict step. Perhaps you want to sample instead?")
