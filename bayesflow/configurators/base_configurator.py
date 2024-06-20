from keras.saving import register_keras_serializable

from bayesflow.types import Tensor


@register_keras_serializable(package="bayesflow.configurators")
class BaseConfigurator:
    @classmethod
    def from_config(cls, config, custom_objects=None) -> "BaseConfigurator":
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

    def configure_inference_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        raise NotImplementedError

    def configure_inference_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        raise NotImplementedError

    def configure_summary_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        raise NotImplementedError

    def configure_summary_conditions(self, data: dict[str, Tensor]) -> Tensor | None:
        raise NotImplementedError
