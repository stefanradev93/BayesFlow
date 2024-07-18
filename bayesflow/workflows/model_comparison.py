from collections.abc import Sequence
import keras

from bayesflow.datasets import OnlineDataset
from bayesflow.simulators import Simulator
from bayesflow.types import Tensor

from .workflow import Workflow


class ModelComparisonApproximator(Workflow):
    """
    Defines an approximator for model (simulator) comparison, where the (discrete) posterior model probabilities are
    learned with a classifier.
    """

    def __init__(
        self,
        classifier: keras.Layer,
        classifier_variables: Sequence[str],
        model_index_name: str = "model_index",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier = classifier
        self.summary_network = ...

    def compute_metrics(self, data: any, stage: str = "training") -> dict[str, Tensor]:
        summary_variables, classifier_variables, model_indices = data

        if self.summary_network is not None:
            summary_outputs = self.summary_network(summary_variables)
            classifier_variables = keras.ops.concatenate([classifier_variables, summary_outputs], axis=-1)

        logits = self.classifier(classifier_variables)

        loss = keras.losses.categorical_crossentropy(model_indices, logits)

        return {"loss": loss}

    def fit(self, simulators: Sequence = None, **kwargs):
        if simulators is None:
            super().fit(**kwargs)

        dataset = OnlineDataset(ModelComparisonSimulator(simulators), ...)

        return super().fit(dataset)


class ModelComparisonSimulator(Simulator):
    def __init__(
        self,
        simulators: Sequence[Simulator],
        p: Sequence[float] = "uniform",
        model_index_sample_fn: str = "uniform",
        **kwargs,
    ):
        self.simulators = ...

    def sample(self, batch_size):
        # draw random model index
        model_index = ...  # randint, or some distribution according to ps
        simulator = self.simulators[model_index]
        # model_index = one_hot(model_index)
        data = simulator.sample(batch_size)

        # TODO: check what name is used in old bayesflow
        return {"model_index": model_index, **data}
