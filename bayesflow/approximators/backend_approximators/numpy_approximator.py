import keras
import numpy as np


class NumpyApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, data: any, stage: str = "training") -> dict[str, np.ndarray]:
        # implemented by each respective architecture
        raise NotImplementedError
