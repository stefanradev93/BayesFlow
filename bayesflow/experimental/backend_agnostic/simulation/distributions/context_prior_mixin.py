
from bayesflow.experimental.backend_agnostic.types import Contexts, Shape, Tensor

from .distribution_mixin import DistributionMixin


class ContextPriorMixin(DistributionMixin):
    """ Base mixin for context priors p(c) """
    def sample(self, batch_shape: Shape) -> Contexts:
        """ Sample contexts: c ~ p(c) """
        raise NotImplementedError

    def log_prob(self, contexts: Contexts) -> Tensor:
        """ Evaluate log p(c) """
        raise NotImplementedError
