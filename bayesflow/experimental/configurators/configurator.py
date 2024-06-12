
from keras import ops

from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import concatenate_tensors

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

    def configure_inference_variables(self, data: dict[str, Tensor]) -> Tensor:
        try:
            data["inference_variables"] = concatenate_tensors(data, self.inference_variables)
        except ValueError as e:
            raise ValueError(f"Cannot trivially concatenate inference variables.") from e

    def configure_inference_conditions(self, data: dict[str, Tensor]) -> Tensor:
        if not self.inference_conditions:
            if "summary_outputs" not in data:
                # case 1: no conditions at all
                return
            else:
                # case 2: just the summaries
                data["inference_conditions"] = data["summary_outputs"]
        else:
            try:
                specified_conditions = concatenate_tensors(data, self.inference_conditions)
            except ValueError as e:
                raise ValueError(f"Cannot trivially concatenate inference conditions.") from e

            if "summary_outputs" not in data:
                # case 3: just the direct inference conditions
                data["inference_conditions"] = specified_conditions
            else:
                # case 4: summaries and direct inference conditions
                try:
                    data["inference_conditions"] = ops.concatenate(
                        [data["summary_outputs"], specified_conditions], axis=-1
                    )
                except ValueError as e:
                    raise ValueError(f"Cannot trivially concatenate summary outputs to inference conditions.") from e

    def configure_summary_variables(self, data: dict[str, Tensor]) -> Tensor:
        if not self.summary_variables:
            return

        try:
            data["summary_variables"] = concatenate_tensors(data, self.summary_variables)
        except ValueError as e:
            raise ValueError(f"Cannot trivially concatenate summary variables along last axis.") from e

    def configure_summary_conditions(self, data: dict[str, Tensor]) -> Tensor:
        if not self.summary_conditions:
            return

        try:
            data["summary_conditions"] = concatenate_tensors(data, self.summary_conditions)
        except ValueError as e:
            raise ValueError(f"Cannot trivially concatenate summary conditions along last axis.") from e
