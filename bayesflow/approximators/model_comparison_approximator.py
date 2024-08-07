from collections.abc import Mapping, Sequence
import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.datasets import OnlineDataset
from bayesflow.data_adapters import ConcatenateKeysDataAdapter, DataAdapter
from bayesflow.networks import SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.types import Shape, Tensor
from bayesflow.utils import filter_kwargs, logging

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

    def build(self, data_shapes: Mapping[str, Shape]):
        data = {key: keras.ops.zeros(value) for key, value in data_shapes.items()}
        self.compute_metrics(data)

    @classmethod
    def build_data_adapter(
        cls,
        classifier_variables: Sequence[str],
        summary_variables: Sequence[str] = None,
        model_index_name: str = "model_indices",
    ):
        variables = {
            "classifier_variables": classifier_variables,
            "summary_variables": summary_variables,
            "model_indices": [model_index_name],
        }
        variables = {key: value for key, value in variables.items() if value is not None}

        return ConcatenateKeysDataAdapter(**variables)

    @classmethod
    def build_dataset(
        cls,
        *,
        dataset: keras.utils.PyDataset = None,
        simulator: ModelComparisonSimulator = None,
        simulators: Sequence[Simulator] = None,
        **kwargs,
    ) -> OnlineDataset:
        if sum(arg is not None for arg in (dataset, simulator, simulators)) != 1:
            raise ValueError("Exactly one of dataset, simulator, or simulators must be provided.")

        if simulators is not None:
            simulator = ModelComparisonSimulator(simulators)

        return super().build_dataset(dataset=dataset, simulator=simulator, **kwargs)

    def compute_metrics(self, data: Mapping[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        classifier_variables = data["classifier_variables"]
        model_indices = data["model_indices"]

        if self.summary_network is not None:
            summary_variables = data["summary_variables"]
            summary_outputs = self.summary_network(summary_variables)

            # TODO: introduce method
            classifier_variables = keras.ops.concatenate([classifier_variables, summary_outputs], axis=-1)

        logits = self.classifier_network(classifier_variables)

        loss = keras.losses.categorical_crossentropy(model_indices, logits)

        return {"loss": loss}

    def fit(
        self,
        *,
        data_adapter: DataAdapter = "auto",
        dataset: keras.utils.PyDataset = None,
        simulator: ModelComparisonSimulator = None,
        simulators: Sequence[Simulator] = None,
        **kwargs,
    ):
        if dataset is not None:
            if simulator is not None or simulators is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, but not both."
                )

            return super().fit(dataset=dataset, **kwargs)

        if data_adapter == "auto":
            logging.info("Building automatic data adapter.")
            data_adapter = self.build_data_adapter(**filter_kwargs(kwargs, self.build_data_adapter))

        if simulator is not None:
            return super().fit(simulator=simulator, data_adapter=data_adapter, **kwargs)

        logging.info(f"Building model comparison simulator from {len(simulators)} simulators.")

        simulator = ModelComparisonSimulator(simulators=simulators)

        return super().fit(simulator=simulator, data_adapter=data_adapter, **kwargs)

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
