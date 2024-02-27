
import keras

from .. import Shape
from ..sampling import SampleLikelihoodMixin
from ..types import Parameters, Contexts, Observations


class AmortizedLikelihood(keras.Model, SampleLikelihoodMixin):
    def __init__(self, surrogate_network: keras.Model) -> None:
        super().__init__()
        self.surrogate_network = surrogate_network

    def sample_likelihood(self, batch_shape: Shape, /, *, parameters: Parameters, contexts: Contexts = None) -> Observations:
        return self.surrogate_network(parameters, contexts)
