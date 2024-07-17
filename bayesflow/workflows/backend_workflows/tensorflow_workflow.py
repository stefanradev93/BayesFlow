import keras
import tensorflow as tf


class TensorFlowWorkflow(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, data: any, stage: str = "training") -> dict[str, tf.Tensor]:
        # implemented by each respective architecture
        raise NotImplementedError

    def train_step(self, data: any, stage: str = "training") -> dict[str, tf.Tensor]:
        raise NotImplementedError
