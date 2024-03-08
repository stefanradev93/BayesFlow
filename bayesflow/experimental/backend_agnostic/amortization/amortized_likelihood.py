
import keras

from bayesflow.experimental.backend_agnostic.simulation import SampleLikelihoodMixin
from bayesflow.experimental.backend_agnostic.types import Data, Parameters, Shape


class AmortizedLikelihood(keras.Model, SampleLikelihoodMixin):
    def __init__(self, surrogate_network: keras.Model):
        super().__init__()
        self.surrogate_network = surrogate_network

    def build(self, input_shape):
        self.surrogate_network.build(input_shape)

    def call(self, *args, **kwargs):
        # we need to override this to use the keras native train loop
        return None

    def compute_loss(self, x: Data = None, *args, **kwargs):
        return self.surrogate_network.compute_loss(x, *args, **kwargs)

    def compute_metrics(self, x: Data = None, *args, **kwargs):
        return self.surrogate_network.compute_metrics(x, *args, **kwargs)

    def sample_likelihood(self, batch_shape: Shape, parameters: Parameters, contexts: Parameters = None) -> Parameters:
        return self.surrogate_network(parameters, contexts)
