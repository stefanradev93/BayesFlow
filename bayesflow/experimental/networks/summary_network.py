
import keras

from bayesflow.experimental.types import Tensor


class SummaryNetwork(keras.Layer):
    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:

        # TODO - find a good way to deal with summary conditions
        data["summary_outputs"] = self(
            data["summary_variables"], training=stage == "training"
        )
        # TODO metrics
        return {}
