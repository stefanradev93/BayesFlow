
import keras

from bayesflow.experimental.backend_agnostic.networks import InferenceNetwork, SummaryNetwork, SurrogateNetwork
from bayesflow.experimental.backend_agnostic.simulation import GenerativeModel, SamplePosteriorMixin, \
    SampleLikelihoodMixin
from bayesflow.experimental.backend_agnostic.types import Data, Observables, Parameters, Shape
from .amortized_likelihood import AmortizedLikelihood
from .amortized_posterior import AmortizedPosterior


class AmortizedPosteriorLikelihood(keras.Model, SamplePosteriorMixin, SampleLikelihoodMixin):
    """ Convenience wrapper for joint amortized posterior and likelihood training """
    def __init__(self, generative_model: GenerativeModel, surrogate_network: SurrogateNetwork, inference_network: InferenceNetwork, summary_network: SummaryNetwork = None):
        super().__init__()
        self.generative_model = generative_model
        self.summary_network = summary_network
        self.inference_network = inference_network
        self.surrogate_network = surrogate_network

        self.amortized_likelihood = AmortizedLikelihood(generative_model=generative_model, surrogate_network=surrogate_network)
        self.amortized_posterior = AmortizedPosterior(generative_model=generative_model, inference_network=inference_network, summary_network=summary_network)

    def build(self, input_shape):
        self.amortized_posterior.build(input_shape)
        self.amortized_likelihood.build(input_shape)

    def call(self, *args, **kwargs):
        # we need to override this to use the keras native train loop
        return None

    def compute_loss(self, x: Data = None, *args, **kwargs):
        posterior_loss = self.amortized_posterior.compute_loss(x, *args, **kwargs)
        likelihood_loss = self.amortized_likelihood.compute_loss(x, *args, **kwargs)

        return posterior_loss + likelihood_loss

    def compute_metrics(self, x: Data = None, *args, **kwargs):
        posterior_metrics = self.amortized_posterior.compute_metrics(x, *args, **kwargs)
        likelihood_metrics = self.amortized_likelihood.compute_metrics(x, *args, **kwargs)

        posterior_metrics = {f"posterior/{key}": value for key, value in posterior_metrics.items()}
        likelihood_metrics = {f"likelihood/{key}": value for key, value in likelihood_metrics.items()}

        metrics = posterior_metrics | likelihood_metrics

        return metrics

    def sample_posterior(self, batch_shape: Shape, data: Data = None) -> Parameters:
        return self.amortized_posterior.sample_posterior(batch_shape, data)

    def sample_likelihood(self, batch_shape: Shape, data: Data = None) -> Observables:
        return self.amortized_likelihood.sample_likelihood(batch_shape, data)
