import keras

from bayesflow.networks import InferenceNetwork, SummaryNetwork

from .workflow import Workflow


class SummaryPlusInference(Workflow):
    """
    Defines a workflow for performing fast posterior or likelihood inference.
    The distribution is approximated with a summary- and an inference network.
    """

    def __init__(self, inference_network: InferenceNetwork, summary_network: SummaryNetwork, **kwargs):
        super().__init__(**kwargs)
        self.inference_network = inference_network
        self.summary_network = summary_network

    def compute_metrics(self, data: any, stage: str = "training"):
        # TODO: data configuration inside dataset
        summary_variables, inference_variables, inference_conditions = data

        # TODO: summary metrics, do not repeat forward pass
        summary_outputs = self.summary_network(summary_variables)

        inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)
        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, stage=stage
        )

        return inference_metrics
