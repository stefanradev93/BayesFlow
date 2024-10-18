from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
from numbers import Number
import numpy as np

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.data_adapters")
class ToArray(ElementwiseTransform):
    def __init__(self):
        super().__init__()
        self.original_type = None

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ToArray":
        instance = cls()
        instance.original_type = deserialize(config["original_type"], custom_objects)
        return instance

    def get_config(self) -> dict:
        return {"original_type": serialize(self.original_type)}

    def forward(self, data: any, **kwargs) -> np.ndarray:
        if self.original_type is None:
            self.original_type = type(data)

        return np.asarray(data)

    def inverse(self, data: np.ndarray, **kwargs) -> any:
        if self.original_type is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        if issubclass(self.original_type, Number):
            try:
                return self.original_type(data.item())
            except ValueError:
                pass

        # cannot invert
        return data
