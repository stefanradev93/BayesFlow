
from .sampling import SampleContextsMixin, SampleParametersMixin, SampleObservationsMixin
from .types import Observations, Shape


class GenerativeModel:
    """ Generate Observations Unconditionally: x ~ p(x) = ∫∫ p(x|θ,c) p(θ|c) p(c) dθ dc """
    def __init__(
            self,
            prior: SampleParametersMixin,
            simulator: SampleObservationsMixin,
            context_prior: SampleContextsMixin = None
    ):
        """
        Parameters
        ----------
        prior : SampleParametersMixin
            The prior to use to generate parameters. See :py:class:`SampleParametersMixin`.
        simulator : SampleObservationsMixin
            The simulator to use to generate observations. See :py:class:`SampleObservationsMixin`.
        context_prior : SampleContextsMixin, optional, default: None
            The context prior to use to generate contexts. See :py:class:`SampleContextsMixin`.
        """
        super().__init__()
        self.context_prior = context_prior
        self.prior = prior
        self.simulator = simulator

    def sample(self, batch_shape: Shape, /) -> Observations:
        if self.context_prior is None:
            contexts = None
        else:
            contexts = self.context_prior.sample_contexts(batch_shape)

        parameters = self.prior.sample_parameters(batch_shape, contexts=contexts)
        observations = self.simulator.sample_observations(batch_shape, parameters=parameters, contexts=contexts)

        return observations
