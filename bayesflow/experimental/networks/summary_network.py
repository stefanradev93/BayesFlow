
import keras

from bayesflow.experimental.types import Tensor


class SummaryNetwork(keras.Layer):
    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        raise NotImplementedError
