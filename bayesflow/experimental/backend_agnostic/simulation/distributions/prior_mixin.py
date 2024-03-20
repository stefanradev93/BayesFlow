
from bayesflow.experimental.backend_agnostic.types import Contexts, Parameters, Shape, Tensor

from .distribution_mixin import DistributionMixin


class PriorMixin(DistributionMixin):
    """ Base mixin for parameter priors p(θ|c) """
    def sample(self, batch_shape: Shape, contexts: Contexts = None) -> Parameters:
        """ Sample parameters: θ ~ p(θ|c) """
        raise NotImplementedError

    def log_prob(self, parameters: Parameters, contexts: Contexts = None) -> Tensor:
        """ Evaluate log p(θ|c) """
        raise NotImplementedError
