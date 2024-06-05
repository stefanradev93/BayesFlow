
import keras

from bayesflow.experimental.types import Tensor


class SummaryNetwork(keras.Layer):
    def compute_loss(self, observed_variables: Tensor, summary_conditions: Tensor = None, **kwargs) -> Tensor:
        return keras.ops.zeros(())

    def compute_metrics(self, observed_variables: Tensor, summary_conditions: Tensor = None, **kwargs) -> dict:
        return {}
