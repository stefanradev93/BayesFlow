from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class Rename(Transform):
    def __init__(self, from_key: str, to_key: str):
        super().__init__()
        self.from_key = from_key
        self.to_key = to_key

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Rename":
        return cls(
            from_key=deserialize(config["from_key"], custom_objects),
            to_key=deserialize(config["to_key"], custom_objects),
        )

    def get_config(self) -> dict:
        return {"from_key": serialize(self.from_key), "to_key": serialize(self.to_key)}

    def forward(self, data: dict[str, any]) -> dict[str, any]:
        data = data.copy()
        data[self.to_key] = data.pop(self.from_key)
        return data

    def inverse(self, data: dict[str, any]) -> dict[str, any]:
        data = data.copy()
        data[self.from_key] = data.pop(self.to_key)
        return data
