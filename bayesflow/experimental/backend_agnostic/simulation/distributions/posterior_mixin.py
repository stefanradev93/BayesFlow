
from bayesflow.experimental.backend_agnostic.types import Contexts, Observables, Parameters, Shape, Tensor

from .distribution_mixin import DistributionMixin


class PosteriorMixin(DistributionMixin):
    """ Base mixin for posteriors p(θ|x,c) """
    def sample(self, batch_shape: Shape, observables: Observables, contexts: Contexts = None) -> Parameters:
        """ Infer parameters from observables: θ ~ p(θ|x,c) """
        raise NotImplementedError

    def log_prob(self, parameters: Parameters, observables: Observables, contexts: Contexts = None) -> Tensor:
        """ Evaluate log p(θ|x,c) """
        raise NotImplementedError
