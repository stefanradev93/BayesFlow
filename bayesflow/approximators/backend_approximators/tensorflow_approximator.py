import keras
import tensorflow as tf


class TensorFlowApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, data: any, stage: str = "training") -> dict[str, tf.Tensor]:
        # implemented by each respective architecture
        raise NotImplementedError

    def test_step(self, data: any) -> dict[str, tf.Tensor]:
        return self.compute_metrics(data, stage="validation")

    def train_step(self, data: any) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            metrics = self.compute_metrics(data)

        loss = metrics["loss"]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._loss_tracker.update_state(loss)

        return metrics
