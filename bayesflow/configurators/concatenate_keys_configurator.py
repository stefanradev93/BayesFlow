from collections.abc import Mapping, Sequence
import keras
from keras.saving import (
    register_keras_serializable as serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import filter_concatenate

from .composite_configurator import CompositeConfigurator
from .configurator import Configurator

DataT = Mapping[str, Tensor]
VarT = Tensor


@serializable(package="bayesflow.configurators")
class _ConcatenateKeysConfigurator(Configurator[DataT, VarT]):
    """Concatenates data from multiple keys into a single tensor."""

    def __init__(self, keys: Sequence[str]):
        if not keys:
            raise ValueError("At least one key must be provided.")

        self.keys = keys
        self.data_shapes = None
        self.is_configured = False

    def configure(self, data: DataT) -> VarT:
        if not self.is_configured:
            self.data_shapes = {key: keras.ops.shape(value) for key, value in data.items()}
            self.is_configured = True

        return filter_concatenate(data, self.keys, axis=-1)

    def deconfigure(self, variables: VarT) -> DataT:
        if not self.is_configured:
            raise ValueError("You must call `configure` at least once before calling `deconfigure`.")

        data = {}
        start = 0
        for key in self.keys:
            stop = start + self.data_shapes[key][-1]
            data[key] = keras.ops.take(variables, list(range(start, stop)), axis=-1)
            start = stop

        return data

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "_ConcatenateKeysConfigurator":
        return cls(config.pop("keys"))

    def get_config(self) -> dict:
        return {"keys": self.keys}


@serializable(package="bayesflow.configurators")
class ConcatenateKeysConfigurator(CompositeConfigurator):
    """Concatenates data from multiple keys into multiple tensors."""

    def __init__(self, **keys: Sequence[str]):
        configurators = {key: _ConcatenateKeysConfigurator(value) for key, value in keys.items()}
        super().__init__(configurators)
