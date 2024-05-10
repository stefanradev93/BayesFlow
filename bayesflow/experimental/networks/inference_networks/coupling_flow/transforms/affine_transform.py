
import numpy as np
from keras import ops as K

from bayesflow.experimental.types import Tensor
from .transform import Transform


class AffineTransform(Transform):
    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        scale, shift = K.split(parameters, 2, axis=-1)

        return {"scale": scale, "shift": shift}

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        shift = np.log(np.e - 1)
        parameters["scale"] = K.softplus(parameters["scale"] + shift)

        return parameters

    def forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        z = parameters["scale"] * x + parameters["shift"]
        log_det = K.mean(K.log(parameters["scale"]), axis=-1)

        return z, log_det

    def inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        x = (z - parameters["shift"]) / parameters["scale"]
        log_det = -K.mean(K.log(parameters["scale"]), axis=-1)

        return x, log_det
