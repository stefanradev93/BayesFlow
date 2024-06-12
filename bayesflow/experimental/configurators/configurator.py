
import keras

from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import filter_concatenate

from .base_configurator import BaseConfigurator


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
