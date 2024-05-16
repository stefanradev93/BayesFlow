
import math

from keras import ops

from bayesflow.experimental.types import Tensor
from .transform import Transform


class AffineTransform(Transform):

    def __init__(self, clamp_factor=5.0, **kwargs):
        super().__init__(**kwargs)
        self.clamp_factor = clamp_factor

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        scale, shift = ops.split(parameters, 2, axis=-1)

        return {"scale": scale, "shift": shift}

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        # shift = math.log(math.e - 1)
        s = parameters["scale"]
        # parameters["scale"] = self.clamp_factor * ops.sigmoid(ops.softplus(parameters["scale"] + shift))
        parameters["scale"] = 1 / (1 + ops.exp(-s)) * ops.sqrt(1 + ops.abs(s + self.clamp_factor))

        return parameters

    def forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        z = parameters["scale"] * x + parameters["shift"]
        log_det = ops.sum(ops.log(parameters["scale"]), axis=-1)

        return z, log_det

    def inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        x = (z - parameters["shift"]) / parameters["scale"]
        log_det = -ops.sum(ops.log(parameters["scale"]), axis=-1)

        return x, log_det
