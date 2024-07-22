from collections.abc import Mapping
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.types import Tensor

from .configurator import Configurator


DataT = Mapping[str, Tensor]
VarT = Mapping[str, Tensor]


@serializable(package="bayesflow.configurators")
class CompositeConfigurator(Configurator[DataT, VarT]):
    """Composes multiple simple configurators into a single more complex configurator."""

    def __init__(self, configurators: Mapping[str, Configurator[DataT, Tensor]]):
        self.configurators = configurators
        self.variable_counts = None

    def configure(self, data: DataT) -> VarT:
        return {key: configurator.configure(data) for key, configurator in self.configurators.items()}

    def deconfigure(self, variables: VarT) -> DataT:
        data = {}
        for key, configurator in self.configurators.items():
            data |= configurator.deconfigure(variables[key])

        return data

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "CompositeConfigurator":
        return cls(
            {
                key: deserialize(configurator, custom_objects)
                for key, configurator in config.pop("configurators").items()
            }
        )

    def get_config(self) -> dict:
        return {"configurators": {key: serialize(configurator) for key, configurator in self.configurators.items()}}
