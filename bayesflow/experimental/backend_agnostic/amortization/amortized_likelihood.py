
import keras

from bayesflow.experimental.backend_agnostic.sampling import SampleLikelihoodMixin
from bayesflow.experimental.backend_agnostic.types import Contexts, Observations, Parameters, Shape


class AmortizedLikelihood(keras.Model, SampleLikelihoodMixin):
    def __init__(self, surrogate_network: keras.Model):
        super().__init__()
        self.surrogate_network = surrogate_network

    def call(self, *args, **kwargs):
        return None
        # TODO: technically this implementation is fine, but it's not very useful
        # parameters, contexts = data["parameters"], data.get("contexts")
        # return self.surrogate_network(parameters, contexts)

    def compute_loss(self, data: dict = None, *args, **kwargs):
        return self.surrogate_network.compute_loss(data, *args, **kwargs)

    def compute_metrics(self, data: dict = None, *args, **kwargs):
        return self.surrogate_network.compute_metrics(data, *args, **kwargs)

    def sample_likelihood(self, batch_shape: Shape, parameters: Parameters, contexts: Contexts = None) -> Observations:
        return self.surrogate_network(parameters, contexts)

    def compute_loss_metrics(self, data):
        if hasattr(self.surrogate_network, "compute_loss_metrics"):
            loss, metrics = self.surrogate_network.compute_loss_metrics(data)
        else:
            loss = self.surrogate_network.compute_loss(data)
            metrics = self.surrogate_network.compute_metrics(data, None, None)

        return loss, metrics
