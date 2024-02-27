
import keras

from .. import Shape
from ..sampling import SamplePosteriorMixin
from ..types import Observations, Contexts, Parameters


class AmortizedPosterior(keras.Model, SamplePosteriorMixin):
    def __init__(self, summary_network: keras.Model, inference_network: keras.Model) -> None:
        super().__init__()
        self.summary_network = summary_network
        self.inference_network = inference_network

    def sample_posterior(self, batch_shape: Shape, /, *, observations: Observations, contexts: Contexts = None) -> Parameters:
        summaries = self.summary_network(observations)
        parameters = self.inference_network(summaries, contexts)
        return parameters
