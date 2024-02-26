
from experimental.backend_agnostic.types import Contexts, Observations, Parameters, Shape


class SamplePosteriorMixin:
    """ Base Mixin for Posterior Sampling """
    def sample_posterior(self, batch_shape: Shape, /, *, observations: Observations, contexts: Contexts = None) -> Parameters:
        """ Parameter Posterior: Infer Parameters from Observations: θ ~ p(θ|x,c) """
        raise NotImplementedError
