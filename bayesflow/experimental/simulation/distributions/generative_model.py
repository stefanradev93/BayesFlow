
from bayesflow.experimental.types import Data, Shape

from .distribution_mixin import DistributionMixin
from .likelihood_mixin import LikelihoodMixin
from .prior_mixin import PriorMixin, UnconditionalPriorMixin


class GenerativeModel(DistributionMixin):
    """ Generate Observables Unconditionally: x ~ p(x) = ∫∫ p(x|θ,c) p(θ|c) p(c) dθ dc """
    is_conditional = False

    def __init__(self, prior: PriorMixin, simulator: LikelihoodMixin, context_prior: UnconditionalPriorMixin = None):
        """
        Parameters
        ----------
        prior : SampleParametersMixin
            The prior to use to generate parameters. See :py:class:`SampleParametersMixin`.
        simulator : SampleObservablesMixin
            The simulator to use to generate observables. See :py:class:`SampleObservablesMixin`.
        context_prior : SampleContextsMixin, optional, default: None
            The context prior to use to generate contexts. See :py:class:`SampleContextsMixin`.
        """
        self.prior = prior
        self.simulator = simulator
        self.context_prior = context_prior

        if self.context_prior is None and self.prior.is_conditional or self.simulator.is_conditional:
            raise ValueError(f"Received a conditional prior or simulator but no conditions.")

    def sample(self, batch_shape: Shape) -> Data:
        if self.context_prior is None:
            contexts = {}
        else:
            contexts = self.context_prior.sample(batch_shape)

        # TODO: this can be improved if contexts is always at least an empty dictionary
        if self.prior.is_conditional:
            parameters = self.prior.sample(batch_shape, contexts)
        else:
            parameters = self.prior.sample(batch_shape)

        if self.simulator.is_conditional:
            observables = self.simulator.sample(batch_shape, parameters, contexts)
        else:
            observables = self.simulator.sample(batch_shape, parameters)

        return Data(contexts=contexts, parameters=parameters, observables=observables)
