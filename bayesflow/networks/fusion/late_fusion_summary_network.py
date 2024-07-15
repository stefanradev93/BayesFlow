import keras

from bayesflow.networks.summary_network import SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs


class LateFusionSummaryNetwork(keras.Layer):
    def __init__(self, summary_networks: dict[str, SummaryNetwork], **kwargs):
        super().__init__(**keras_kwargs(kwargs))

        self.num_data_sources = len(summary_networks)
        self.summary_networks = summary_networks

    def build(self, input_shape, **kwargs):
        for summary_network in self.summary_networks.values():
            summary_network.build(input_shape, **kwargs)

    def call(self, x: Tensor, **kwargs) -> Tensor:
        """
        :param x: Tensor of shape (batch_size, set_size, input_dim)

        :param kwargs: Additional keyword arguments.

        :return: Tensor of shape (batch_size, output_dim)
        """
        outputs = [] * self.num_data_sources
        # Pass all data sources through their respective summary network
        for i, (source_name, summary_network) in enumerate(self.summary_networks.items()):
            data_source = {"summary_variables": x[source_name]}
            outputs[i] = summary_network(data_source, training=kwargs.get("training", False))

        # Concatenate the outputs of the individual summary networks
        return keras.ops.concatenate(outputs, axis=-1)

    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        metrics_sources = {}

        summary_variables = data["summary_variables"]

        # Pass all data sources through their respective summary network
        for source_name, summary_network in self.summary_networks.items():
            data_source = {"summary_variables": summary_variables[source_name]}
            metrics_sources[source_name] = summary_network.compute_metrics(data_source, training=stage == "training")

        # Merge all information (outputs, loss, additional metrics)
        metrics_out = {}

        # fuse (concatenate) the outputs of the individual summary networks
        try:
            outputs = [metrics["outputs"] for metrics in metrics_sources.values()]
            metrics_out["outputs"] = keras.ops.concatenate(outputs, axis=-1)
        except ValueError as e:
            shapes = [metrics["outputs"].shape for metrics in metrics_sources.values()]
            raise ValueError(f"Cannot trivially concatenate outputs with shapes {shapes}") from e

        # sum up any losses of the individual summary networks
        metrics_out["loss"] = keras.ops.sum([metrics["loss"] for metrics in metrics_sources.values()], axis=0)

        # gather remaining metrics (only relevant if not training)
        if stage != "training":
            for source_name, source_metrics in metrics_sources.items():
                for metric_name, metric_value in source_metrics.items():
                    if metric_name not in ["loss", "outputs"]:
                        metrics_out[f"{source_name}_{metric_name}"] = metric_value

        return metrics_out
