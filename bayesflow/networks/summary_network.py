import keras

from bayesflow.types import Tensor


class SummaryNetwork(keras.Layer):
    def call(self, data: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        summary_inputs = data["summary_variables"]
        if data.get("summary_conditions") is not None:
            summary_inputs = keras.ops.concatenate([summary_inputs, data["summary_conditions"]], axis=-1)

        outputs = self(summary_inputs, training=stage == "training")

        if any(self.metrics):
            # TODO: what should we do here?
            raise NotImplementedError

        return {"loss": keras.ops.zeros(()), "outputs": outputs}
