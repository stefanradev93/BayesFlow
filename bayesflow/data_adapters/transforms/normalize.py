from keras.saving import (
    register_keras_serializable as serializable,
)
import numpy as np

from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class Normalize(Transform):
    """Normalizes a parameter to have zero mean and unit standard deviation"""

    def __init__(self, parameter_name: str, mean: np.ndarray, std: np.ndarray):
        super().__init__(parameter_name)
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)

    @classmethod
    def from_batch(cls, batch: dict[str, np.ndarray], parameter_name: str) -> "Normalize":
        parameter = batch[parameter_name]
        mean = np.mean(parameter, axis=0)
        std = np.std(parameter, axis=0)

        return cls(parameter_name, mean, std)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Normalize":
        return cls(config["parameter_name"], np.array(config["mean"]), np.array(config["std"]))

    def forward(self, parameter: np.ndarray) -> np.ndarray:
        return (parameter - self.mean) / self.std

    def get_config(self) -> dict:
        return {
            "parameter_name": self.parameter_name,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def inverse(self, parameter: np.ndarray) -> np.ndarray:
        return parameter * self.std + self.mean
