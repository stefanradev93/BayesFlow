from collections.abc import Mapping, Sequence
import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.configurators import ConcatenateKeysConfigurator, Configurator
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
        inference_network: InferenceNetwork,
        summary_network: SummaryNetwork = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inference_network = inference_network
        self.summary_network = summary_network

    def build(self, data_shapes: Mapping[str, Shape]):
        data = {key: keras.ops.zeros(value) for key, value in data_shapes.items()}
        self.compute_metrics(data)

    def compute_metrics(self, data: Mapping[str, Tensor], stage: str = "training"):
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
        configurator: Configurator = "auto",
        inference_variables: Sequence[str] = None,
        inference_conditions: Sequence[str] = None,
        summary_variables: Sequence[str] = None,
        **kwargs,
    ):
        if "dataset" in kwargs:
            return super().fit(**kwargs)

        if configurator == "auto":
            logging.info("Building automatic configurator.")
            configurator = ConcatenateKeysConfigurator(
                inference_variables=inference_variables,
                inference_conditions=inference_conditions,
                summary_variables=summary_variables,
            )

        return super().fit(configurator=configurator, **kwargs)

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
