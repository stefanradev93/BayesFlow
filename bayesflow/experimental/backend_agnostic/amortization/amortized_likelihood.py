
import keras

from bayesflow.experimental.backend_agnostic.simulation import GenerativeModel, SampleLikelihoodMixin
from bayesflow.experimental.backend_agnostic.types import Data, Observables, Shape
from bayesflow.experimental.backend_agnostic.utils import nested_merge


class AmortizedLikelihood(keras.Model, SampleLikelihoodMixin):
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

    def sample_likelihood(self, batch_shape: Shape, data: Data = None, surrogate_kwargs: dict = None) -> Observables:
        if data is None:
            data = self.generative_model.sample(batch_shape)
        else:
            data = nested_merge(data, self.generative_model.sample(batch_shape))

        surrogate_kwargs = surrogate_kwargs or {}

        return {"observables": self.surrogate_network.sample(batch_shape, data, **surrogate_kwargs)}
