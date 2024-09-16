from collections.abc import Sequence
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class Normalize(Transform):
    """Normalizes a parameter to have zero mean and unit standard deviation.
    By default, this is lazily initialized; the mean and standard deviation are computed from the first batch of data.
    For eager initialization, pass the mean and standard deviation to the constructor.
    """

    def __init__(
        self, parameters: str | Sequence[str] | None = None, /, *, mean: np.ndarray = None, std: np.ndarray = None
    ):
        super().__init__(parameters)
        self.mean = mean
        self.std = std

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Normalize":
        return cls(
            deserialize(config["parameters"], custom_objects),
            mean=deserialize(config["mean"], custom_objects),
            std=deserialize(config["std"], custom_objects),
        )

    def forward(self, parameter: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            self.mean = np.mean(parameter, axis=0)
            self.std = np.std(parameter, axis=0)

        return (parameter - self.mean) / self.std

    def get_config(self) -> dict:
        return {
            "parameters": serialize(self.parameters),
            "mean": serialize(self.mean),
            "std": serialize(self.std),
        }

    def inverse(self, parameter: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Cannot call `inverse` before calling `forward` at least once.")

        return parameter * self.std + self.mean
