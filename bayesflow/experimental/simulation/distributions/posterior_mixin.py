
from bayesflow.experimental.types import Contexts, Observables, Parameters, Shape, Tensor

from bayesflow.experimental.simulation.distributions.distribution_mixin import DistributionMixin


class ConditionalPosteriorMixin(DistributionMixin):
    """ Base mixin for posteriors p(θ|x,c) """
    is_conditional = True

    def sample(self, batch_shape: Shape, observables: Observables, contexts: Contexts) -> Parameters:
        """ Infer parameters from observables: θ ~ p(θ|x,c) """
        raise NotImplementedError

    def log_prob(self, parameters: Parameters, observables: Observables, contexts: Contexts) -> Tensor:
        """ Evaluate log p(θ|x,c) """
        raise NotImplementedError


class UnconditionalPosteriorMixin(DistributionMixin):
    """ Base mixin for posteriors p(θ|x) """
    is_conditional = False

    def sample(self, batch_shape: Shape, observables: Observables) -> Parameters:
        """ Infer parameters from observables: θ ~ p(θ|x) """
        raise NotImplementedError

    def log_prob(self, parameters: Parameters, observables: Observables) -> Tensor:
        """ Evaluate log p(θ|x) """
        raise NotImplementedError


PosteriorMixin = ConditionalPosteriorMixin | UnconditionalPosteriorMixin
