
from bayesflow.experimental.backend_agnostic.types import Data, Shape
from .contexts import SampleContextsMixin
from .observables import SampleObservablesMixin
from .parameters import SampleParametersMixin


class GenerativeModel:
    """ Generate Observables Unconditionally: x ~ p(x) = ∫∫ p(x|θ,c) p(θ|c) p(c) dθ dc """
    def __init__(
            self,
            prior: SampleParametersMixin,
            simulator: SampleObservablesMixin,
            context_prior: SampleContextsMixin = None
    ):
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
        self.context_prior = context_prior
        self.prior = prior
        self.simulator = simulator

    def sample(self, batch_shape: Shape) -> Data:
        if self.context_prior is None:
            contexts = None
        else:
            contexts = self.context_prior.sample_contexts(batch_shape)

        parameters = self.prior.sample_parameters(batch_shape, contexts=contexts)
        observables = self.simulator.sample_observables(batch_shape, parameters=parameters, contexts=contexts)

        return {"contexts": contexts, "parameters": parameters, "observables": observables}
