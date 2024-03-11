
from bayesflow.experimental.backend_agnostic.types import Data, Observables, Shape


class SampleLikelihoodMixin:
    """ Base Mixin for Likelihood Sampling """
    def sample_likelihood(self, batch_shape: Shape, data: Data = None) -> Observables:
        """ Surrogate Simulator: Infer Observables from Parameters: x ~ p(x|θ,c) """
        # user-implemented
        raise NotImplementedError
