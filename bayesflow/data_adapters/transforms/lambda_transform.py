from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class LambdaTransform(Transform):
    """
    Transforms a parameter using a pair of forward and inverse functions.

    Important note: This class is only serializable if the forward and inverse functions are serializable.
    This most likely means you will have to pass the scope that the forward and inverse functions are contained in
    to the `custom_objects` argument of the `deserialize` function when deserializing this class.
    """

    def __init__(self, parameter_name: str, forward: callable, inverse: callable):
        super().__init__(parameter_name)

        self.forward = forward
        self.inverse = inverse

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "LambdaTransform":
        return cls(
            deserialize(config["parameter_name"], custom_objects),
            deserialize(config["forward"], custom_objects),
            deserialize(config["inverse"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "parameter_name": serialize(self.parameter_name),
            "forward": serialize(self.forward),
            "inverse": serialize(self.inverse),
        }
