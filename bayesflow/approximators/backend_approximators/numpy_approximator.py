import keras
import numpy as np

from bayesflow.utils import filter_kwargs


class NumpyApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, np.ndarray]:
        # implemented by each respective architecture
        raise NotImplementedError

    def test_step(self, data: dict[str, any]) -> dict[str, np.ndarray]:
        kwargs = filter_kwargs(data | {"stage": "validation"}, self.compute_metrics)
        return self.compute_metrics(**kwargs)

    def train_step(self, data: dict[str, any]) -> dict[str, np.ndarray]:
        raise NotImplementedError("Numpy backend does not support training.")
