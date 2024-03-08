
import keras

from bayesflow.experimental.backend_agnostic.simulation import SamplePosteriorMixin
from bayesflow.experimental.backend_agnostic.types import Contexts, Observables, Parameters, Shape
from bayesflow.experimental.backend_agnostic.types import Data


class AmortizedPosterior(keras.Model, SamplePosteriorMixin):
    def __init__(self, inference_network: keras.Model, summary_network: keras.Model = None):
        super().__init__()
        self.inference_network = inference_network
        self.summary_network = summary_network

    def build(self, input_shape):
        if self.summary_network is not None:
            self.summary_network.build(input_shape)

        self.inference_network.build(input_shape)

    def call(self, *args, **kwargs):
        # we need to override this to use the keras native train loop
        return None

    def compute_loss(self, x: Data = None, *args, **kwargs):
        if self.summary_network is not None:
            if "summaries" not in x:
                x["summaries"] = self.summary_network(x)

            summary_loss = self.summary_network.compute_loss(x, *args, **kwargs)
            inference_loss = self.inference_network.compute_loss(x, *args, **kwargs)

            return summary_loss + inference_loss

        return self.inference_network.compute_loss(x, *args, **kwargs)

    def compute_metrics(self, x: Data = None, *args, **kwargs):
        if self.summary_network is not None:
            if "summaries" not in x:
                x["summaries"] = self.summary_network(x)

            summary_metrics = self.summary_network.compute_metrics(x, *args, **kwargs)
            inference_metrics = self.inference_network.compute_metrics(x, *args, **kwargs)

            summary_metrics = {f"summary/{key}": value for key, value in summary_metrics.items()}
            inference_metrics = {f"inference/{key}": value for key, value in inference_metrics.items()}

            return summary_metrics | inference_metrics

        return self.inference_network.compute_metrics(x, *args, **kwargs)

    def sample_posterior(self, batch_shape: Shape, observables: Observables, contexts: Contexts = None) -> Parameters:
        data = {"observables": observables, "contexts": contexts}
        if self.summary_network is not None:
            data["summaries"] = self.summary_network(data)

        return self.inference_network(data)
