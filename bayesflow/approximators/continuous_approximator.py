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

    def build(self, data_shapes: Mapping[str, Shape]) -> None:
        data = {key: keras.ops.zeros(value) for key, value in data_shapes.items()}
        self.compute_metrics(**data, stage="training")

    @classmethod
    def build_data_adapter(
        cls,
        inference_variables: Sequence[str],
        inference_conditions: Sequence[str],
        summary_variables: Sequence[str] = None,
    ) -> DataAdapter:
        variables = {
            "inference_variables": inference_variables,
            "inference_conditions": inference_conditions,
            "summary_variables": summary_variables,
        }
        variables = {key: value for key, value in variables.items() if value is not None}

        return ConcatenateKeysDataAdapter(**variables)

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        if self.summary_network is not None:
            summary_outputs = self.summary_network(summary_variables)

            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, stage=stage
        )

        return inference_metrics

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inference_network = deserialize(config.pop("inference_network"), custom_objects=custom_objects)
        summary_network = deserialize(config.pop("summary_network"), custom_objects=custom_objects)

        return cls(inference_network=inference_network, summary_network=summary_network, **config)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "inference_network": serialize(self.inference_network),
            "summary_network": serialize(self.summary_network),
        }

        return base_config | config

    def sample(
        self,
        batch_shape: Shape,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        numpy: bool = False,
    ) -> dict[str, Tensor]:
        num_datasets, num_samples = batch_shape

        if self.summary_network is not None:
            # TODO: get data from user input and configure into summary variables
            summary_outputs = self.summary_network(summary_variables)

            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                # TODO: use data adapter
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        # inference_variables = self.inference_network.sample(batch_shape, conditions=inference_conditions)

        # TODO: populate dictionary with param_name: param_value pairs
        samples = ...

        if numpy:
            samples = {key: keras.ops.convert_to_numpy(value) for key, value in samples.items()}

        return samples

    def log_prob(self, data: Mapping[str, Tensor], numpy: bool = False) -> Tensor:
        data = self.data_adapter.configure(data)
        inference_variables = data["inference_variables"]
        inference_conditions = data.get("inference_conditions")

        if self.summary_network is not None:
            summary_variables = data["summary_variables"]
            summary_outputs = self.summary_network(summary_variables)

            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                # TODO: use data adapter
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        log_prob = self.inference_network.log_prob(inference_variables, conditions=inference_conditions)

        if numpy:
            log_prob = keras.ops.convert_to_numpy(log_prob)

        return log_prob
