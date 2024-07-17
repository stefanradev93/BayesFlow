import keras

from bayesflow.types import Tensor

from .workflow import Workflow


class ModelComparison(Workflow):
    """
    Defines a workflow for simulator comparison, where a discrete ... (posterior?) is learned with a classifier
    """

    def __init__(self, classifier: keras.Layer, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier

    def compute_metrics(self, data: any, stage: str = "training") -> dict[str, Tensor]:
        # TODO: data configuration inside dataset
        x, y_true = data
        y_pred = self.classifier(x)

        loss = keras.losses.categorical_crossentropy(y_true, y_pred)

        return {"loss": loss}
