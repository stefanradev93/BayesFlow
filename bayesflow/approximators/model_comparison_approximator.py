from collections.abc import Sequence
import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.configurators import ConcatenateKeysConfigurator, Configurator
from bayesflow.networks import SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.types import Shape, Tensor
from bayesflow.utils import logging

from .approximator import Approximator


@serializable(package="bayesflow.approximators")
class ModelComparisonApproximator(Approximator):
    """
    Defines an approximator for model (simulator) comparison, where the (discrete) posterior model probabilities are
    learned with a classifier.
    """

    def __init__(
        self,
        *,
        classifier_network: keras.Layer,
        summary_network: SummaryNetwork = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier_network = classifier_network
        self.summary_network = summary_network

    def build(self, data_shapes: Sequence[Shape]):
        data = []
        for shape in data_shapes:
            if shape is None:
                data.append(None)
            else:
                data.append(keras.ops.zeros(shape))

        self.compute_metrics(data)

    def compute_metrics(self, data: any, stage: str = "training") -> dict[str, Tensor]:
        classifier_variables, summary_variables, model_indices = data

        if self.summary_network is not None:
            summary_outputs = self.summary_network(summary_variables)
            classifier_variables = keras.ops.concatenate([classifier_variables, summary_outputs], axis=-1)

        logits = self.classifier_network(classifier_variables)

        loss = keras.losses.categorical_crossentropy(model_indices, logits)

        return {"loss": loss}

    def fit(
        self,
        *,
        configurator: Configurator = "auto",
        classifier_variables: Sequence[str] = None,
        model_index_name: str = "model_indices",
        simulators: Sequence[Simulator] = None,
        summary_variables: Sequence[str] = None,
        **kwargs,
    ):
        if "dataset" in kwargs:
            return super().fit(**kwargs)

        if configurator == "auto":
            logging.info("Building automatic configurator.")
            configurator = ConcatenateKeysConfigurator([classifier_variables, summary_variables, [model_index_name]])

        if "simulator" in kwargs:
            return super().fit(configurator=configurator, **kwargs)

        logging.info("Building model comparison simulator from {n} simulators.", n=len(simulators))

        simulator = ModelComparisonSimulator(simulators=simulators)

        return super().fit(simulator=simulator, configurator=configurator, **kwargs)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        classifier_network = deserialize(config.pop("classifier_network"), custom_objects=custom_objects)
        summary_network = deserialize(config.pop("summary_network"), custom_objects=custom_objects)
        return cls(classifier_network=classifier_network, summary_network=summary_network, **config)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "classifier_network": serialize(self.classifier_network),
            "summary_network": serialize(self.summary_network),
        }

        return base_config | config
