from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class LambdaTransform(Transform):
    def __init__(self, parameter_name: str, forward: callable, inverse: callable):
        super().__init__(parameter_name)

        self.forward = forward
        self.inverse = inverse

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "LambdaTransform":
        return cls(config["parameter_name"], deserialize(config["forward"]), deserialize(config["inverse"]))

    def get_config(self) -> dict:
        return {
            "parameter_name": self.parameter_name,
            "forward": serialize(self.forward),
            "inverse": serialize(self.inverse),
        }
