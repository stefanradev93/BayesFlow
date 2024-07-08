import keras
from keras.saving import register_keras_serializable


from bayesflow.types import Tensor
from bayesflow.utils import filter_concatenate

from .base_configurator import BaseConfigurator


@register_keras_serializable(package="bayesflow.configurators")
class Configurator(BaseConfigurator):
    def __init__(
        self,
        inference_variables: list[str],
        inference_conditions: list[str] = None,
        summary_variables: list[str] = None,
    ):
        self.inference_variables = inference_variables
        self.inference_conditions = inference_conditions or []
        self.summary_variables = summary_variables or []

        self._inference_indices = None

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Configurator":
        inference_indices = config.pop("inference_indices")
        instance = cls(**config)
        instance._inference_indices = inference_indices

        return instance

    def get_config(self) -> dict:
        return {
            "inference_variables": self.inference_variables,
            "inference_conditions": self.inference_conditions,
            "summary_variables": self.summary_variables,
            "inference_indices": self._inference_indices,
        }

    def configure_inference_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        if self._inference_indices is None:
            inference_variables = {key: value for key, value in data.items() if key in self.inference_variables}

            self._inference_indices = {}
            start = 0
            for key, value in inference_variables.items():
                stop = start + keras.ops.shape(value)[-1]
                self._inference_indices[key] = list(range(start, stop))
                start = stop

        return filter_concatenate(data, keys=self.inference_variables)

    def configure_inference_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        keys = self.inference_conditions
        if "summary_outputs" in data and "summary_outputs" not in keys:
            keys = keys.copy()
            keys.append("summary_outputs")

        return filter_concatenate(data, keys=keys)

    def configure_summary_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        return filter_concatenate(data, keys=self.summary_variables)

    def deconfigure(self, data: Tensor) -> dict[str, Tensor]:
        return {key: keras.ops.take(data, index, axis=-1) for key, index in self._inference_indices.items()}
