import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)
import warnings

from bayesflow.configurators import BaseConfigurator
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Shape, Tensor
from bayesflow.utils import keras_kwargs


@register_keras_serializable(package="bayesflow.approximators")
class BaseApproximator(keras.Model):
    def __init__(
        self,
        inference_network: InferenceNetwork,
        summary_network: SummaryNetwork,
        configurator: BaseConfigurator,
        **kwargs,
    ):
        super().__init__(**keras_kwargs(kwargs))
        self.inference_network = inference_network
        self.summary_network = summary_network
        self.configurator = configurator

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "BaseApproximator":
        config["inference_network"] = deserialize_keras_object(
            config["inference_network"], custom_objects=custom_objects
        )
        config["summary_network"] = deserialize_keras_object(config["summary_network"], custom_objects=custom_objects)
        config["configurator"] = deserialize_keras_object(config["configurator"], custom_objects=custom_objects)

        return cls(**config)

    def get_config(self) -> dict:
        base_config = super().get_config()

        config = {
            "inference_network": serialize_keras_object(self.inference_network),
            "summary_network": serialize_keras_object(self.summary_network),
            "configurator": serialize_keras_object(self.configurator),
        }

        return base_config | config

    # noinspect PyMethodOverriding
    def build(self, data_shapes: dict[str, Shape]):
        data = {name: keras.ops.zeros(shape) for name, shape in data_shapes.items()}
        self.build_from_data(data)

    def build_from_data(self, data: dict[str, Tensor]):
        self.compute_metrics(data, stage="training")
        self.built = True

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
            warnings.warn(
                "Found no validation logs due to a bug in keras. "
                "Applying workaround, but incorrect loss values may be logged. "
                "If possible, increase the size of your dataset, "
                "or lower the number of validation steps used."
            )

            val_logs = {}

        return val_logs

    # noinspection PyMethodOverriding
    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        # compiled modes do not allow in-place operations on the data object
        # we perform a shallow copy here, which is cheap
        data = data.copy()

        if self.summary_network is None:
            data["inference_variables"] = self.configurator.configure_inference_variables(data)
            data["inference_conditions"] = self.configurator.configure_inference_conditions(data)
            return self.inference_network.compute_metrics(data, stage=stage)

        data["summary_variables"] = self.configurator.configure_summary_variables(data)
        data["summary_conditions"] = self.configurator.configure_summary_conditions(data)

        summary_metrics = self.summary_network.compute_metrics(data, stage=stage)

        data["summary_outputs"] = summary_metrics.pop("outputs")

        data["inference_variables"] = self.configurator.configure_inference_variables(data)
        data["inference_conditions"] = self.configurator.configure_inference_conditions(data)

        inference_metrics = self.inference_network.compute_metrics(data, stage=stage)

        metrics = {"loss": summary_metrics["loss"] + inference_metrics["loss"]}

        summary_metrics = {f"summary/{key}": val for key, val in summary_metrics.items()}
        inference_metrics = {f"inference/{key}": val for key, val in inference_metrics.items()}

        return metrics | summary_metrics | inference_metrics

    def compute_loss(self, *args, **kwargs):
        raise RuntimeError("Use compute_metrics()['loss'] instead.")

    def fit(self, *args, **kwargs):
        if not self.built:
            try:
                dataset = kwargs.get("x") or args[0]
                self.build_from_data(dataset[0])
            except Exception:
                raise RuntimeError(
                    "Could not automatically build the approximator. Please pass a dataset as the "
                    "first argument to `approximator.fit()` or manually call `approximator.build()` "
                    "with a dictionary specifying your data shapes."
                )

        return super().fit(*args, **kwargs)
