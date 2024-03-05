
import keras

from bayesflow.experimental.backend_agnostic.sampling import SamplePosteriorMixin
from bayesflow.experimental.backend_agnostic.types import Contexts, Observations, Parameters, Shape


class AmortizedPosterior(keras.Model, SamplePosteriorMixin):
    def __init__(self, summary_network: keras.Model, inference_network: keras.Model):
        super().__init__()
        self.summary_network = summary_network
        self.inference_network = inference_network

    def call(self, *args, **kwargs):
        # TODO: can we implement this?
        return None

        # observations, parameters = data["observations"], data["parameters"]
        # if "summaries" in data:
        #     summaries = data["summaries"]
        # else:
        #     summaries = self.summary_network(observations)
        #
        # TODO: not valid because inference_network is not required to be callable or yield predictions here
        # return self.inference_network(summaries, parameters)

    def compute_loss(self, data: dict = None, *args, **kwargs):
        observations, parameters = data["observations"], data["parameters"]

        if "summaries" not in data:
            data["summaries"] = self.summary_network(observations)

        summary_loss = self.summary_network.compute_loss(data, *args, **kwargs)
        inference_loss = self.inference_network.compute_loss(data, *args, **kwargs)

        return summary_loss + inference_loss

    def compute_metrics(self, data: dict = None, *args, **kwargs):
        observations, parameters = data["observations"], data["parameters"]

        if "summaries" not in data:
            data["summaries"] = self.summary_network(observations)

        summary_metrics = self.summary_network.compute_metrics(data, *args, **kwargs)
        inference_metrics = self.inference_network.compute_metrics(data, *args, **kwargs)

        summary_metrics = {f"summary/{key}": value for key, value in summary_metrics.items()}
        inference_metrics = {f"inference/{key}": value for key, value in inference_metrics.items()}

        metrics = summary_metrics | inference_metrics

        return metrics

    def sample_posterior(self, batch_shape: Shape, observations: Observations, contexts: Contexts = None) -> Parameters:
        summaries = self.summary_network(observations)
        parameters = self.inference_network(summaries, contexts)
        return parameters

    def compute_loss_metrics(self, data):
        observations, parameters = data["observations"], data["parameters"]

        data["summaries"] = self.summary_network(observations)

        if hasattr(self.summary_network, "compute_loss_metrics"):
            summary_loss, summary_metrics = self.summary_network.compute_loss_metrics(data)
        else:
            summary_loss = self.summary_network.compute_loss(data)
            summary_metrics = self.summary_network.compute_metrics(data, None, None)

        if hasattr(self.inference_network, "compute_loss_metrics"):
            inference_loss, inference_metrics = self.inference_network.compute_loss_metrics(data)
        else:
            inference_loss = self.inference_network.compute_loss(data)
            inference_metrics = self.inference_network.compute_metrics(data, None, None)

        summary_metrics = {f"summary/{key}": value for key, value in summary_metrics.items()}
        inference_metrics = {f"inference/{key}": value for key, value in inference_metrics.items()}

        loss = summary_loss + inference_loss
        metrics = summary_metrics | inference_metrics

        return loss, metrics
