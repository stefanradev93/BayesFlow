from collections.abc import Sequence
import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.configurators import ConcatenateKeysConfigurator, Configurator
from bayesflow.networks import InferenceNetwork, SummaryNetwork
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

    def build(self, data_shapes):
        data = []
        for shape in data_shapes:
            if shape is None:
                data.append(None)
            else:
                data.append(keras.ops.zeros(shape))

        self.compute_metrics(data)

    def compute_metrics(self, data: any, stage: str = "training"):
        inference_variables, summary_variables, inference_conditions = data

        if self.summary_network is None:
            return self.inference_network.compute_metrics(
                inference_variables, conditions=inference_conditions, stage=stage
            )

        summary_outputs = self.summary_network(summary_variables)

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
            configurator = ConcatenateKeysConfigurator([inference_variables, summary_variables, inference_conditions])

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
