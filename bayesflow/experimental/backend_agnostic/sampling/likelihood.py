
from ..types import Contexts, Observations, Parameters, Shape


class SampleLikelihoodMixin:
    """ Base Mixin for Likelihood Sampling """
    def sample_likelihood(self, batch_shape: Shape, parameters: Parameters, contexts: Contexts = None) -> Observations:
        """ Surrogate Simulator: Infer Observations from Parameters: x ~ p(x|θ,c) """
        # user-implemented
        raise NotImplementedError
