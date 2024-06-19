
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
        summary_conditions: list[str] = None
    ):
        self.inference_variables = inference_variables
        self.inference_conditions = inference_conditions or []
        self.summary_variables = summary_variables or []
        self.summary_conditions = summary_conditions or []

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Configurator":
        return cls(**config)

    def get_config(self) -> dict:
        return {
            "inference_variables": self.inference_variables,
            "inference_conditions": self.inference_conditions,
            "summary_variables": self.summary_variables,
            "summary_conditions": self.summary_conditions
        }

    def configure_inference_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        return filter_concatenate(data, keys=self.inference_variables)

    def configure_inference_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        keys = self.inference_conditions.copy()
        if "summary_outputs" in data and "summary_outputs" not in keys:
            keys.append("summary_outputs")

        return filter_concatenate(data, keys=keys)

    def configure_summary_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        return filter_concatenate(data, keys=self.summary_variables)

    def configure_summary_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        return filter_concatenate(data, keys=self.summary_conditions)
