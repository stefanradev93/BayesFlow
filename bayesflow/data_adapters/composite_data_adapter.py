from collections.abc import Mapping
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.types import Tensor

from .data_adapter import DataAdapter


TRaw = Mapping[str, Tensor]
TReady = Mapping[str, Tensor]


@serializable(package="bayesflow.configurators")
class CompositeDataAdapter(DataAdapter[TRaw, TReady]):
    """Composes multiple simple data adapters into a single more complex adapter."""

    def __init__(self, configurators: Mapping[str, DataAdapter[TRaw, Tensor]]):
        self.configurators = configurators
        self.variable_counts = None

    def configure(self, data: TRaw) -> TReady:
        return {key: configurator.configure(data) for key, configurator in self.configurators.items()}

    def deconfigure(self, variables: TReady) -> TRaw:
        data = {}
        for key, configurator in self.configurators.items():
            data |= configurator.deconfigure(variables[key])

        return data

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "CompositeDataAdapter":
        return cls(
            {
                key: deserialize(configurator, custom_objects)
                for key, configurator in config.pop("configurators").items()
            }
        )

    def get_config(self) -> dict:
        return {"configurators": {key: serialize(configurator) for key, configurator in self.configurators.items()}}
