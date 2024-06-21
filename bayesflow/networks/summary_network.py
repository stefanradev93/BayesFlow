import keras

from bayesflow.types import Tensor


class SummaryNetwork(keras.Layer):
    def call(self, data: dict[str, Tensor], stage: str = "training") -> Tensor:
        raise NotImplementedError

    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        outputs = self(data, stage=stage)

        return {"loss": keras.ops.zeros(()), "outputs": outputs}
