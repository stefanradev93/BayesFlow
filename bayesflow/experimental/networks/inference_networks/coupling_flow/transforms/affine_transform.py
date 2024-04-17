
import keras
import numpy as np

from bayesflow.experimental.types import Tensor
from .transform import Transform


class AffineTransform(Transform):
    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        scale, shift = keras.ops.split(parameters, 2, axis=1)

        return {"scale": scale, "shift": shift}

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        shift = np.log(np.e - 1)
        parameters["scale"] = keras.ops.softplus(parameters["scale"] + shift)

        return parameters

    def forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        z = parameters["scale"] * x + parameters["shift"]
        logdet = keras.ops.mean(keras.ops.log(parameters["scale"]), axis=1)

        return z, logdet

    def inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        x = (z - parameters["shift"]) / parameters["scale"]
        logdet = -keras.ops.mean(keras.ops.log(parameters["scale"]), axis=1)

        return x, logdet
