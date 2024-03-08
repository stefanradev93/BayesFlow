
import keras

from bayesflow.experimental.backend_agnostic.simulation import SamplePosteriorMixin, SampleLikelihoodMixin
from bayesflow.experimental.backend_agnostic.types import Contexts, Data, Observables, Parameters, Shape
from .amortized_likelihood import AmortizedLikelihood
from .amortized_posterior import AmortizedPosterior


class AmortizedPosteriorLikelihood(keras.Model, SamplePosteriorMixin, SampleLikelihoodMixin):
    """ Convenience wrapper for joint amortized posterior and likelihood training """
    def __init__(self, surrogate_network: keras.Model, inference_network: keras.Model, summary_network: keras.Model = None):
        super().__init__()
        self.summary_network = summary_network
        self.inference_network = inference_network
        self.surrogate_network = surrogate_network

        self.amortized_likelihood = AmortizedLikelihood(surrogate_network=surrogate_network)
        self.amortized_posterior = AmortizedPosterior(inference_network=inference_network, summary_network=summary_network)

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

    def sample_posterior(self, batch_shape: Shape, observables: Observables, contexts: Contexts = None) -> Parameters:
        return self.amortized_posterior.sample_posterior(batch_shape, observables, contexts)

    def sample_likelihood(self, batch_shape: Shape, parameters: Parameters, contexts: Contexts = None) -> Observables:
        return self.amortized_likelihood.sample_likelihood(batch_shape, parameters, contexts)
