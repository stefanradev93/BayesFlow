from keras.saving import register_keras_serializable

from bayesflow.types import Tensor

from .configurator import Configurator


@register_keras_serializable(package="bayesflow.configurators")
class DictConfigurator(Configurator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_summary_variables(self, data: dict[str, Tensor]) -> dict[str, Tensor] | None:
        return {k: v for k, v in data.items() if k in self.summary_variables}
