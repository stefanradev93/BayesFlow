
from experimental.backend_agnostic.types import Contexts, Observations, Parameters, Shape


class SampleObservationsMixin:
    """ Base Mixin for Observation Sampling """
    def sample_observations(self, batch_shape: Shape, /, *, parameters: Parameters, contexts: Contexts = None) -> Observations:
        """ Simulator: Sample Observations given Parameters: x ~ p(x|θ,c) """
        raise NotImplementedError
