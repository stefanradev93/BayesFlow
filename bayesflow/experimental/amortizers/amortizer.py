
import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Tensor

from .base_amortizer import BaseAmortizer


@register_keras_serializable(package="bayesflow.amortizers")
class Amortizer(BaseAmortizer):
    def __init__(self, inferred_variables: list[str], observed_variables: list[str], inference_conditions: list[str] = None, summary_conditions: list[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.inferred_variables = inferred_variables
        self.observed_variables = observed_variables
        self.inference_conditions = inference_conditions or []
        self.summary_conditions = summary_conditions or []

    def configure_inferred_variables(self, data: dict[str, Tensor]) -> Tensor:
        return keras.ops.concatenate([data[key] for key in self.inferred_variables])

    def configure_observed_variables(self, data: dict[str, Tensor]) -> Tensor:
        return keras.ops.concatenate([data[key] for key in self.observed_variables])

    def configure_inference_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        if not self.inference_conditions:
            return None

        return keras.ops.concatenate([data[key] for key in self.inference_conditions])

    def configure_summary_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        if not self.summary_conditions:
            return None

        return keras.ops.concatenate([data[key] for key in self.summary_conditions])
