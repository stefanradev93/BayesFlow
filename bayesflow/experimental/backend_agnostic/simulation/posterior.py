
from bayesflow.experimental.backend_agnostic.types import Data, Parameters, Shape


class SamplePosteriorMixin:
    """ Base Mixin for Posterior Sampling """
    def sample_posterior(self, batch_shape: Shape, data: Data = None) -> Parameters:
        """ Parameter Posterior: Infer Parameters from Observables: θ ~ p(θ|x,c) """
        # user-implemented
        raise NotImplementedError
