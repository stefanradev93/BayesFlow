
import keras

from bayesflow.experimental.backend_agnostic.sampling import SampleLikelihoodMixin, SamplePosteriorMixin
from bayesflow.experimental.backend_agnostic.types import Contexts, Observations, Parameters, Shape
from .amortized_likelihood import AmortizedLikelihood
from .amortized_posterior import AmortizedPosterior


class AmortizedPosteriorLikelihood(keras.Model, SamplePosteriorMixin, SampleLikelihoodMixin):
    """ Convenience wrapper for joint amortized posterior and likelihood training """
    def __init__(self, summary_network: keras.Model, inference_network: keras.Model, surrogate_network: keras.Model):
        super().__init__()
        self.summary_network = summary_network
        self.inference_network = inference_network
        self.surrogate_network = surrogate_network

        self.amortized_likelihood = AmortizedLikelihood(surrogate_network)
        self.amortized_posterior = AmortizedPosterior(summary_network, inference_network)

    def compute_loss_metrics(self, data):
        posterior_loss, posterior_metrics = self.amortized_posterior.compute_loss_metrics(data)
        likelihood_loss, likelihood_metrics = self.amortized_likelihood.compute_loss_metrics(data)

        posterior_metrics = {f"posterior/{key}": value for key, value in posterior_metrics.items()}
        likelihood_metrics = {f"likelihood/{key}": value for key, value in likelihood_metrics.items()}

        metrics = posterior_metrics | likelihood_metrics

        return posterior_loss + likelihood_loss, metrics

    def sample_posterior(self, batch_shape: Shape, observations: Observations, contexts: Contexts = None) -> Parameters:
        summaries = self.summary_network(observations)
        parameters = self.inference_network(summaries, contexts)
        return parameters

    def sample_likelihood(self, batch_shape: Shape, parameters: Parameters, contexts: Contexts = None) -> Observations:
        return self.surrogate_network(parameters, contexts)
