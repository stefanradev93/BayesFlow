
import keras

from bayesflow.experimental.backend_agnostic.sampling import SamplePosteriorMixin
from bayesflow.experimental.backend_agnostic.types import Contexts, Observations, Parameters, Shape


class AmortizedPosterior(keras.Model, SamplePosteriorMixin):
    def __init__(self, summary_network: keras.Model, inference_network: keras.Model):
        super().__init__()
        self.summary_network = summary_network
        self.inference_network = inference_network

    def sample_posterior(self, batch_shape: Shape, observations: Observations, contexts: Contexts = None) -> Parameters:
        summaries = self.summary_network(observations)
        parameters = self.inference_network(summaries, contexts)
        return parameters

    def compute_loss_metrics(self, data):
        observations, parameters = data["observations"], data["parameters"]

        summaries = self.summary_network(observations)
        summary_loss = self.summary_network.compute_loss(summaries)
        inference_loss = self.inference_network.compute_loss(summaries, parameters)

        return summary_loss + inference_loss
