
from bayesflow.experimental.backend_agnostic.types import Data, Shape

from .distributions import ContextDistributionMixin as CDM, ObservableDistributionMixin as ODM, ParameterDistributionMixin as PDM


class GenerativeModel:
    """ Generate Observables Unconditionally: x ~ p(x) = ∫∫ p(x|θ,c) p(θ|c) p(c) dθ dc """
    def __init__(self, parameter_distribution: PDM, observable_distribution: ODM, context_distribution: CDM = None):
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
        super().__init__()
        self.context_distribution = context_distribution
        self.observable_distribution = observable_distribution
        self.parameter_distribution = parameter_distribution

    def sample(self, batch_shape: Shape) -> Data:
        if self.context_prior is None:
            contexts = None
        else:
            contexts = self.context_prior.sample_contexts(batch_shape)

        parameters = self.prior.sample_parameters(batch_shape, contexts=contexts)
        observables = self.simulator.sample_observables(batch_shape, parameters=parameters, contexts=contexts)

        return {"contexts": contexts, "parameters": parameters, "observables": observables}
