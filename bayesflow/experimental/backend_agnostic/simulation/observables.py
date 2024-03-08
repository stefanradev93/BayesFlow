
from bayesflow.experimental.backend_agnostic.types import Contexts, Parameters, Observables, Shape


class SampleObservablesMixin:
    """ Base Mixin for Observables Sampling """
    def sample_observables(self, batch_shape: Shape, parameters: Parameters, contexts: Contexts = None) -> Observables:
        """ Simulator: Sample Observables given Parameters: x ~ p(x|θ,c) """
        # user-implemented
        raise NotImplementedError
