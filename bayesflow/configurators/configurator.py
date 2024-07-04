import keras
from keras.saving import register_keras_serializable

import numpy as np

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

        self._splits = None

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Configurator":
        splits = config.pop("splits")
        instance = cls(**config)
        instance._splits = splits

        return instance

    def get_config(self) -> dict:
        return {
            "inference_variables": self.inference_variables,
            "inference_conditions": self.inference_conditions,
            "summary_variables": self.summary_variables,
            "splits": self._splits,
        }

    def configure_inference_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        if self._splits is None:
            inference_variables = [value for key, value in data.items() if key in self.inference_variables]
            splits = [keras.ops.shape(x)[-1] for x in inference_variables]
            self._splits = np.cumsum(splits).tolist()

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
        splits = keras.ops.split(data, self._splits, axis=-1)
        # drop empty last split
        splits = splits[:-1]
        return dict(zip(self.inference_variables, splits))
