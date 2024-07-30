from collections.abc import Mapping, Sequence
import keras
from keras.saving import (
    register_keras_serializable as serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import filter_concatenate

from .composite_data_adapter import CompositeDataAdapter
from .data_adapter import DataAdapter

TRaw = Mapping[str, Tensor]
TProcessed = Tensor


@serializable(package="bayesflow.configurators")
class _ConcatenateKeysDataAdapter(DataAdapter[TRaw, TProcessed]):
    """Concatenates data from multiple keys into a single tensor."""

    def __init__(self, keys: Sequence[str]):
        if not keys:
            raise ValueError("At least one key must be provided.")

        self.keys = keys
        self.data_shapes = None
        self.is_configured = False

    def configure(self, raw_data: TRaw) -> TProcessed:
        if not self.is_configured:
            self.data_shapes = {key: keras.ops.shape(value) for key, value in raw_data.items()}
            self.is_configured = True

        return filter_concatenate(raw_data, self.keys, axis=-1)

    def deconfigure(self, processed_data: TProcessed) -> TRaw:
        if not self.is_configured:
            raise ValueError("You must call `configure` at least once before calling `deconfigure`.")

        data = {}
        start = 0
        for key in self.keys:
            stop = start + self.data_shapes[key][-1]
            data[key] = keras.ops.take(processed_data, list(range(start, stop)), axis=-1)
            start = stop

        return data

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "_ConcatenateKeysDataAdapter":
        return cls(config.pop("keys"))

    def get_config(self) -> dict:
        return {"keys": self.keys}


@serializable(package="bayesflow.configurators")
class ConcatenateKeysDataAdapter(CompositeDataAdapter):
    """Concatenates data from multiple keys into multiple tensors."""

    def __init__(self, **keys: Sequence[str]):
        configurators = {key: _ConcatenateKeysDataAdapter(value) for key, value in keys.items()}
        super().__init__(configurators)
