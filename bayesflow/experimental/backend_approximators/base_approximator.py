
import keras

from bayesflow.experimental.types import Tensor


class BaseApproximator(keras.Model):
    def train_step(self, data):
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def compute_metrics(self, data: dict[str, Tensor], mode: str = "training") -> Tensor:
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError(f"Use compute_metrics()['loss'] instead.")
