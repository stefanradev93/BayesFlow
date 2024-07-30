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
from bayesflow.utils import filter_kwargs, logging

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
        inference_network: InferenceNetwork,
        summary_network: SummaryNetwork = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inference_network = inference_network
        self.summary_network = summary_network

    def build(self, data_shapes: Mapping[str, Shape]) -> None:
        data = {key: keras.ops.zeros(value) for key, value in data_shapes.items()}
        self.compute_metrics(data)

    def build_data_adapter(
        self,
        inference_variables: Sequence[str],
        inference_conditions: Sequence[str],
        summary_variables: Sequence[str] = None,
    ) -> DataAdapter:  # TODO: generic types
        variables = {
            "inference_variables": inference_variables,
            "inference_conditions": inference_conditions,
            "summary_variables": summary_variables,
        }
        variables = {key: value for key, value in variables.items() if value is not None}

        return ConcatenateKeysDataAdapter(**variables)

    def compute_metrics(self, data: Mapping[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        # TODO: add method or property to return required keys, on top of documentation
        inference_variables = data["inference_variables"]
        inference_conditions = data.get("inference_conditions")

        if self.summary_network is not None:
            summary_variables = data["summary_variables"]
            summary_outputs = self.summary_network(summary_variables)

            # TODO: introduce method
            if inference_conditions is not None:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, stage=stage
        )

        return inference_metrics

    def fit(
        self,
        *,
        data_adapter: DataAdapter = "auto",
        dataset: keras.utils.PyDataset = None,
        **kwargs,
    ):
        if dataset is not None:
            return super().fit(dataset=dataset, **kwargs)

        if data_adapter == "auto":
            logging.info("Building automatic data adapter.")
            data_adapter = self.build_data_adapter(**filter_kwargs(kwargs, self.build_data_adapter))

        return super().fit(data_adapter=data_adapter, **kwargs)

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

    def sample(self, batch_shape: Shape, conditions: Tensor = None, numpy: bool = False) -> dict[str, Tensor]:
        ...
        # num_datasets, num_samples = batch_shape
        #
        # if self.summary_network is not None:
        #     summary_outputs = self.summary_network(data["summary_variables"])
        #
        #     inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)
        #
        # samples = self.inference_network.sample(batch_shape, conditions=conditions)
        #
        # if numpy:
        #     samples = {key: keras.ops.convert_to_numpy(value) for key, value in samples.items()}
        #
        # return samples
        #
        # if self.summary_network is not None:
        #     summary_outputs = self.summary_network(data["summary_variables"])
        #     data["summary_outputs"] = self.summary_network(data["summary_variables"])
        #
        # inference_conditions = self.configurator.configure_inference_conditions(data)
        #
        # # TODO: do not assume this is a tensor
        # # TODO: do not rely on ndim == 2 vs ndim == 3 (i.e., allow multiple feature dimensions for conditions)
        # if inference_conditions is not None and keras.ops.ndim(inference_conditions) == 2:
        #     inference_conditions = expand_tile(inference_conditions, axis=1, n=num_samples)
        #
        # samples = self.inference_network.sample(batch_shape, conditions=inference_conditions)
        # samples = self.configurator.deconfigure(samples)
        #
        # if self.summary_network is not None:
        #     samples["summaries"] = data["summary_outputs"]
        #
        # return process_output(samples, convert_to_numpy=numpy)

    def log_prob(self, data: Mapping[str, Tensor], numpy: bool = False) -> Tensor:
        ...
        # data = data.copy()
        #
        # if self.summary_network is not None:
        #     data["summary_variables"] = self.configurator.configure_summary_variables(data)
        #     data["summary_outputs"] = self.summary_network(data["summary_variables"])
        #
        # data["inference_conditions"] = self.configurator.configure_inference_conditions(data)
        # data["inference_variables"] = self.configurator.configure_inference_variables(data)
        #
        # log_prob = self.inference_network.log_prob(data["inference_variables"],
        # conditions=data["inference_conditions"])
        #
        # if numpy:
        #     log_prob = keras.ops.convert_to_numpy(log_prob)
        #
        # return log_prob
