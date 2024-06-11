
import keras
import warnings

from bayesflow.experimental.configurators import BaseConfigurator
from bayesflow.experimental.networks import InferenceNetwork, SummaryNetwork
from bayesflow.experimental.types import Shape, Tensor


class BaseApproximator(keras.Model):
    def __init__(self, inference_network: InferenceNetwork, summary_network: SummaryNetwork, configurator: BaseConfigurator, **kwargs):
        super().__init__(**kwargs)
        self.inference_network = inference_network
        self.summary_network = summary_network
        self.configurator = configurator

    # noinspect PyMethodOverriding
    def build(self, data_shapes: dict[str, Shape]):
        data = {name: keras.ops.zeros(shape) for name, shape in data_shapes.items()}
        self.compute_metrics(data, stage="training")

    def train_step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        # we cannot provide a backend-agnostic implementation due to reliance on autograd
        raise NotImplementedError

    def test_step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics = self.compute_metrics(data, stage="validation")
        self._loss_tracker.update_state(metrics["loss"])
        return metrics

    def evaluate(self, *args, **kwargs):
        val_logs = super().evaluate(*args, **kwargs)

        if val_logs is None:
            # https://github.com/keras-team/keras/issues/19835
            warnings.warn(f"Found no validation logs due to a bug in keras. "
                          f"Applying workaround, but incorrect loss values may be logged. "
                          f"If possible, increase the size of your dataset, "
                          f"or lower the number of validation steps used.")

            val_logs = {}

        return val_logs

    # noinspection PyMethodOverriding
    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        if self.summary_network is None:
            self.configurator.configure_inference_variables(data)
            self.configurator.configure_inference_conditions(data)

            return self.inference_network.compute_metrics(data, stage=stage)

        self.configurator.configure_summary_variables(data)
        self.configurator.configure_summary_conditions(data)
        summary_metrics = self.summary_network.compute_metrics(data, stage=stage)

        self.configurator.configure_inference_variables(data)
        self.configurator.configure_inference_conditions(data)
        inference_metrics = self.inference_network.compute_metrics(data, stage=stage)

        metrics = {"loss": summary_metrics["loss"] + inference_metrics["loss"]}

        summary_metrics = {f"summary/{key}": val for key, val in summary_metrics.items()}
        inference_metrics = {f"inference/{key}": val for key, val in inference_metrics.items()}

        return metrics | summary_metrics | inference_metrics

    def compute_loss(self, *args, **kwargs):
        raise RuntimeError(f"Use compute_metrics()['loss'] instead.")

    def fit(self, *args, **kwargs):
        try:
            dataset = kwargs.get("x") or args[0]
            test_batch = dataset[0]

            data_shapes = {key: keras.ops.shape(val) for key, val in test_batch.items()}
            self.build(data_shapes)
        except Exception:
            raise RuntimeError(f"Could not automatically build the approximator. Please pass a dataset as the first "
                               f"argument to `approximator.fit()` or manually call `approximator.build()` with a "
                               f"dictionary specifying your data shapes.")

        return super().fit(*args, **kwargs)
