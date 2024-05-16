
from math import pi as PI_CONST

from keras import ops

from bayesflow.experimental.types import Tensor
from .transform import Transform


class AffineTransform(Transform):

    def __init__(self, clamp_factor=1.9):
        self.clamp_factor = clamp_factor

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        scale, shift = ops.split(parameters, 2, axis=-1)

        return {"scale": scale, "shift": shift}

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        s = (2.0 * self.clamp_factor / PI_CONST) * ops.atan(parameters["scale"] / self.clamp_factor)
        parameters["scale"] = ops.exp(s)

        return parameters

    def forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        z = parameters["scale"] * x + parameters["shift"]
        log_det = ops.mean(parameters["scale"], axis=-1)

        return z, log_det

    def inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        x = (z - parameters["shift"]) / parameters["scale"]
        log_det = -ops.mean(parameters["scale"], axis=-1)

        return x, log_det
