import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable,
    serialize_keras_object as serialize,
)

from bayesflow.networks import InferenceNetwork, SummaryNetwork

from .workflow import Workflow


@register_keras_serializable(package="bayesflow.workflows")
class ContinuousApproximator(Workflow):
    """
    Defines a workflow for performing fast posterior or likelihood inference.
    The distribution is approximated with an inference network and an optional summary network.
    """

    def __init__(self, inference_network: InferenceNetwork, summary_network: SummaryNetwork = None, **kwargs):
        super().__init__(**kwargs)
        self.inference_network = inference_network
        self.summary_network = summary_network

    def build(self, data_shapes):
        data = [keras.ops.zeros(shape) for shape in data_shapes]
        self.compute_metrics(data)

    def compute_metrics(self, data: any, stage: str = "training"):
        summary_variables, inference_variables, inference_conditions = data

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

    def fit(self, *, collate_fn: callable = None, **kwargs):
        if collate_fn is None:
            collate_fn = ...
        return super().fit(collate_fn=collate_fn, **kwargs)

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
