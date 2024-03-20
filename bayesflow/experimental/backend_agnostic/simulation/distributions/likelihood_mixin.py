
from bayesflow.experimental.backend_agnostic.types import Contexts, Observables, Parameters, Shape, Tensor

from .distribution_mixin import DistributionMixin

# TODO: bayesflow dev chat -> observables / likelihood? only one


class LikelihoodMixin(DistributionMixin):
    """ Base mixin for simulators and surrogate simulators p(x|θ,c) """
    def sample(self, batch_shape: Shape, parameters: Parameters, contexts: Contexts = None) -> Observables:
        """ Infer observables from parameters: x ~ p(x|θ,c) """
        raise NotImplementedError

    def log_prob(self, observables: Observables, parameters: Parameters, contexts: Contexts = None) -> Tensor:
        """ Evaluate log p(x|θ,c) """
        raise NotImplementedError
