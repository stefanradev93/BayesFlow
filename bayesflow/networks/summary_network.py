import keras

from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor
from bayesflow.utils import find_distribution, keras_kwargs


class SummaryNetwork(keras.Layer):
    def __init_(self, base_distribution: str = "normal", **kwargs):
        super().__init__(**keras_kwargs(kwargs))

        self.base_distribution = find_distribution(base_distribution)

    def build(self, input_shape, **kwargs):
        if self.base_distribution is not None:
            input_shape = self.compute_output_shape(input_shape)
            self.base_distribution.build(input_shape)

    def call(self, x: Tensor, **kwargs) -> Tensor:
        """
        :param x: Tensor of shape (batch_size, set_size, input_dim)

        :param kwargs: Additional keyword arguments.

        :return: Tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError

    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        x = data["summary_variables"]
        outputs = self(x, training=stage == "training")

        metrics = {}

        if self.base_distribution is not None:
            samples = self.base_distribution.sample((keras.ops.shape(x)[0],))
            mmd = maximum_mean_discrepancy(outputs, samples)
            metrics["loss"] = keras.ops.mean(mmd)

            if stage != "training":
                # compute sample-based validation metrics
                for metric in self.metrics:
                    metrics[metric.name] = metric(outputs, samples)
        else:
            metrics["loss"] = keras.ops.zeros(())

        metrics["outputs"] = outputs

        return metrics


class FusionSummaryNetwork(keras.Layer):
    def __init__(self, summary_networks: list[SummaryNetwork], **kwargs):
        super().__init__(**keras_kwargs(kwargs))

        self.summary_networks = summary_networks
