from keras.saving import register_keras_serializable

from bayesflow.types import Tensor
from bayesflow.utils import filter_tuple

from .configurator import Configurator


@register_keras_serializable(package="bayesflow.configurators")
class TupleConfigurator(Configurator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_summary_variables(self, data: dict[str, Tensor]) -> Tensor | None:
        return filter_tuple(data, keys=self.summary_variables)
