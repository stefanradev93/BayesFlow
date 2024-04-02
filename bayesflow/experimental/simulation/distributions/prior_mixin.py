
from bayesflow.experimental.types import Contexts, Parameters, Shape, Tensor

from bayesflow.experimental.simulation.distributions.distribution_mixin import DistributionMixin


class ConditionalPriorMixin(DistributionMixin):
    """ Base mixin for priors p(θ|c) """
    is_conditional = True

    def sample(self, batch_shape: Shape, contexts: Contexts) -> Parameters:
        """ Sample parameters: θ ~ p(θ|c) """
        raise NotImplementedError

    def log_prob(self, parameters: Parameters, contexts: Contexts) -> Tensor:
        """ Evaluate log p(θ|c) """
        raise NotImplementedError


class UnconditionalPriorMixin(DistributionMixin):
    """ Base mixin for priors p(θ) """
    is_conditional = False

    def sample(self, batch_shape: Shape) -> Parameters:
        """ Sample parameters: θ ~ p(θ) """
        raise NotImplementedError

    def log_prob(self, parameters: Parameters) -> Tensor:
        """ Evaluate log p(θ) """
        raise NotImplementedError


PriorMixin = ConditionalPriorMixin | UnconditionalPriorMixin
