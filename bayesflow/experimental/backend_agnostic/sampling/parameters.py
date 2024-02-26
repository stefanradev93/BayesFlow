
from experimental.backend_agnostic.types import Contexts, Parameters, Shape


class SampleParametersMixin:
    """ Base Mixin for Parameter Sampling """
    def sample_parameters(self, batch_shape: Shape, /, *, contexts: Contexts = None) -> Parameters:
        """ Parameter Prior: Sample Parameters: θ ~ p(θ|c) """
        raise NotImplementedError
