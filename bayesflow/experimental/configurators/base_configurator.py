
from bayesflow.experimental.types import Tensor


class BaseConfigurator:
    def configure_inference_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        raise NotImplementedError

    def configure_inference_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        raise NotImplementedError

    def configure_summary_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        raise NotImplementedError

    def configure_summary_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        raise NotImplementedError
