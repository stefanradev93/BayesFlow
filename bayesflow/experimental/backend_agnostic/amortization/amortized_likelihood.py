
import keras

from bayesflow.experimental.backend_agnostic.simulation import GenerativeModel
from bayesflow.experimental.backend_agnostic.simulation.distributions import LikelihoodDistributionMixin
from bayesflow.experimental.backend_agnostic.types import Data, Observables, Shape
from bayesflow.experimental.backend_agnostic.utils import nested_merge


class AmortizedLikelihood(keras.Model, LikelihoodDistributionMixin):
    def __init__(self, generative_model: GenerativeModel, surrogate_network: keras.Model):
        super().__init__()
        self.generative_model = generative_model
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

    def sample(self, *args, **kwargs):
        return self.surrogate_network.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.surrogate_network.log_prob(*args, **kwargs)
