
from bayesflow.experimental.backend_agnostic.types import Contexts, Shape


class SampleContextsMixin:
    """ Base Mixin for Context Sampling """
    def sample_contexts(self, batch_shape: Shape) -> Contexts:
        """ Context Prior: Sample Context: c ~ p(c) """
        # user-implemented
        raise NotImplementedError
