from bayesflow.networks import InferenceNetwork

from .workflow import Workflow


class JustInference(Workflow):
    """
    Defines a workflow for performing fast posterior or likelihood inference.
    The distribution is approximated with an inference network.
    """

    def __init__(self, inference_network: InferenceNetwork, **kwargs):
        super().__init__(**kwargs)
        self.inference_network = inference_network

    def compute_metrics(self, data: any, stage: str = "training"):
        # TODO: data configuration inside dataset
        x, conditions = data
        return self.inference_network.compute_metrics(x, conditions=conditions, stage=stage)
