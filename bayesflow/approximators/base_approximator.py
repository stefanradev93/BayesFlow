import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)
from collections.abc import Sequence
import warnings

from bayesflow.configurators import BaseConfigurator
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Shape, Tensor
from bayesflow.utils import keras_kwargs, expand_tile, process_output


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

    def sample(self, batch_shape: Shape, data: dict[str, Tensor] = None, numpy: bool = False) -> dict[str, Tensor]:
        num_datasets, num_samples = batch_shape

        if data is None:
            data = {}
        else:
            data = data.copy()

        if self.summary_network is not None:
            data["summary_variables"] = self.configurator.configure_summary_variables(data)
            data["summary_outputs"] = self.summary_network(data["summary_variables"])

        inference_conditions = self.configurator.configure_inference_conditions(data)

        # TODO: do not assume this is a tensor
        # TODO: do not rely on ndim == 2 vs ndim == 3 (i.e., allow multiple feature dimensions for conditions)
        if inference_conditions is not None and keras.ops.ndim(inference_conditions) == 2:
            inference_conditions = expand_tile(inference_conditions, axis=1, n=num_samples)

        samples = self.inference_network.sample(batch_shape, conditions=inference_conditions)
        samples = self.configurator.deconfigure(samples)

        if self.summary_network is not None:
            samples["summaries"] = data["summary_outputs"]

        return process_output(samples, convert_to_numpy=numpy)

    def log_prob(self, data: dict[str, Tensor], numpy: bool = False) -> Tensor:
        data = data.copy()

        if self.summary_network is not None:
            data["summary_variables"] = self.configurator.configure_summary_variables(data)
            data["summary_outputs"] = self.summary_network(data["summary_variables"])

        data["inference_conditions"] = self.configurator.configure_inference_conditions(data)
        data["inference_variables"] = self.configurator.configure_inference_variables(data)

        log_prob = self.inference_network.log_prob(data["inference_variables"], conditions=data["inference_conditions"])

        if numpy:
            log_prob = keras.ops.convert_to_numpy(log_prob)

        return log_prob

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
                "Found no validation logs due to a bug in keras. Applying workaround, but incorrect loss values may be "
                "logged. If possible, increase the size of your dataset, or lower the number of validation steps used."
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
            except IndexError:
                raise RuntimeError("Missing fit data.")

            if not isinstance(dataset, keras.utils.PyDataset):
                raise RuntimeError(
                    "Cannot automatically build the approximator. Please pass a dataset as the "
                    "first argument to `approximator.fit()` or manually call `approximator.build()` "
                    "with a dictionary specifying your data shapes."
                )

            data = next(iter(dataset))
            self.build_from_data(data)

        return super().fit(*args, **kwargs)

    def compile(
        self, inference_metrics: Sequence[keras.Metric] = None, summary_metrics: Sequence[keras.Metric] = None, **kwargs
    ) -> None:
        if inference_metrics:
            self.inference_network._metrics = inference_metrics

        if summary_metrics:
            if self.summary_network is None:
                warnings.warn("Ignoring summary metrics because there is no summary network.")
            else:
                self.summary_network._metrics = summary_metrics

        return super().compile(**kwargs)
