
from bayesflow.experimental.backend_agnostic.types import Contexts, Observables, Parameters, Shape


class SamplePosteriorMixin:
    """ Base Mixin for Posterior Sampling """
    def sample_posterior(self, batch_shape: Shape, observables: Observables, contexts: Contexts = None) -> Parameters:
        """ Parameter Posterior: Infer Parameters from Observables: θ ~ p(θ|x,c) """
        # user-implemented
        raise NotImplementedError
