
from bayesflow.experimental.types import Contexts, Observables, Parameters, Shape, Tensor

from bayesflow.experimental.simulation.distributions.distribution_mixin import DistributionMixin

# TODO: bayesflow dev chat -> observables / likelihood? only one


class ConditionalLikelihoodMixin(DistributionMixin):
    """ Base mixin for simulators p(x|θ,c) """
    is_conditional = True

    def sample(self, batch_shape: Shape, parameters: Parameters, contexts: Contexts) -> Observables:
        """ Infer observables from parameters: x ~ p(x|θ,c) """
        raise NotImplementedError

    def log_prob(self, observables: Observables, parameters: Parameters, contexts: Contexts) -> Tensor:
        """ Evaluate log p(x|θ,c) """
        raise NotImplementedError


class UnconditionalLikelihoodMixin(DistributionMixin):
    """ Base mixin for simulators p(x|θ) """
    is_conditional = False

    def sample(self, batch_shape: Shape, parameters: Parameters) -> Observables:
        """ Infer observables from parameters: x ~ p(x|θ) """
        raise NotImplementedError

    def log_prob(self, observables: Observables, parameters: Parameters) -> Tensor:
        """ Evaluate log p(x|θ) """
        raise NotImplementedError


LikelihoodMixin = ConditionalLikelihoodMixin | UnconditionalLikelihoodMixin
