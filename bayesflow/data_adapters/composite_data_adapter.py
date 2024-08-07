from collections.abc import Mapping
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .data_adapter import DataAdapter


TRaw = Mapping[str, np.ndarray]
TProcessed = Mapping[str, np.ndarray]


@serializable(package="bayesflow.data_adapters")
class CompositeDataAdapter(DataAdapter[TRaw, TProcessed]):
    """Composes multiple simple data adapters into a single more complex adapter."""

    def __init__(self, data_adapters: Mapping[str, DataAdapter[TRaw, np.ndarray]]):
        self.data_adapters = data_adapters
        self.variable_counts = None

    def configure(self, data: TRaw) -> TProcessed:
        return {key: data_adapter.configure(data) for key, data_adapter in self.data_adapters.items()}

    def deconfigure(self, variables: TProcessed) -> TRaw:
        data = {}
        for key, data_adapter in self.data_adapters.items():
            data |= data_adapter.deconfigure(variables[key])

        return data

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "CompositeDataAdapter":
        return cls(
            {
                key: deserialize(data_adapter, custom_objects)
                for key, data_adapter in config.pop("data_adapters").items()
            }
        )

    def get_config(self) -> dict:
        return {"data_adapters": {key: serialize(configurator) for key, configurator in self.data_adapters.items()}}
