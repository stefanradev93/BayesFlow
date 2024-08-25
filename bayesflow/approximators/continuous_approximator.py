from collections.abc import Mapping, Sequence
import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.data_adapters import ConcatenateKeysDataAdapter, DataAdapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Shape, Tensor
from bayesflow.utils import logging

from .approximator import Approximator


@serializable(package="bayesflow.approximators")
class ContinuousApproximator(Approximator):
    """
    Defines a workflow for performing fast posterior or likelihood inference.
    The distribution is approximated with an inference network and an optional summary network.
    """

    def __init__(
        self,
        *,
        data_adapter: DataAdapter,
        inference_network: InferenceNetwork,
        summary_network: SummaryNetwork = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_adapter = data_adapter
        self.inference_network = inference_network
        self.summary_network = summary_network

    @classmethod
    def build_data_adapter(
        cls,
        inference_variables: Sequence[str],
        inference_conditions: Sequence[str] = None,
        summary_variables: Sequence[str] = None,
    ) -> DataAdapter:
        variables = {
            "inference_variables": inference_variables,
            "inference_conditions": inference_conditions,
            "summary_variables": summary_variables,
        }
        variables = {key: value for key, value in variables.items() if value is not None}

        return ConcatenateKeysDataAdapter(**variables)

    def compile(
        self,
        *args,
        inference_metrics: Sequence[keras.Metric] = None,
        summary_metrics: Sequence[keras.Metric] = None,
        **kwargs,
    ):
        if inference_metrics:
            self.inference_network._metrics = inference_metrics

        if summary_metrics:
            if self.summary_network is None:
                logging.warning("Ignoring summary metrics because there is no summary network.")
            else:
                self.summary_network._metrics = summary_metrics

        return super().compile(*args, **kwargs)

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        if self.summary_network is None:
            summary_metrics = {}
        else:
            summary_metrics = self.summary_network.compute_metrics(summary_variables, stage=stage)
            summary_outputs = summary_metrics.pop("outputs")

            # append summary outputs to inference conditions
            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, stage=stage
        )

        loss = inference_metrics.get("loss", keras.ops.zeros(())) + summary_metrics.get("loss", keras.ops.zeros(()))

        inference_metrics = {f"{key}/inference_{key}": value for key, value in inference_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | inference_metrics | summary_metrics

        return metrics

    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs, data_adapter=self.data_adapter)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["data_adapter"] = deserialize(config["data_adapter"], custom_objects=custom_objects)
        config["inference_network"] = deserialize(config["inference_network"], custom_objects=custom_objects)
        config["summary_network"] = deserialize(config["summary_network"], custom_objects=custom_objects)

        return super().from_config(config, custom_objects=custom_objects)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "data_adapter": serialize(self.data_adapter),
            "inference_network": serialize(self.inference_network),
            "summary_network": serialize(self.summary_network),
        }

        return base_config | config

    def sample(self, batch_shape: Shape, data: Mapping[str, Tensor], numpy: bool = False) -> dict[str, Tensor]:
        data = self.data_adapter.configure(data)
        data = keras.tree.map_structure(keras.ops.convert_to_tensor, data)
        data = {"inference_variables": self._sample(batch_shape, **data)}
        data = self.data_adapter.deconfigure(data)

        if numpy:
            data = keras.tree.map_structure(keras.ops.convert_to_numpy, data)

        return data

    def _sample(
        self, batch_shape: Shape, inference_conditions: Tensor = None, summary_variables: Tensor = None
    ) -> Tensor:
        if self.summary_network is not None:
            summary_outputs = self.summary_network(summary_variables)

            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        return self.inference_network.sample(batch_shape, conditions=inference_conditions)

    def log_prob(self, data: Mapping[str, Tensor], numpy: bool = False) -> Tensor:
        data = self.data_adapter.configure(data)
        log_prob = self._log_prob(**data)

        if numpy:
            log_prob = keras.ops.convert_to_numpy(log_prob)

        return log_prob

    def _log_prob(
        self, inference_variables: Tensor, inference_conditions: Tensor = None, summary_variables: Tensor = None
    ) -> Tensor:
        if self.summary_network is not None:
            summary_outputs = self.summary_network(summary_variables)

            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        return self.inference_network.log_prob(inference_variables, conditions=inference_conditions)
