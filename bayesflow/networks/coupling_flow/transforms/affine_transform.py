import keras
import keras.ops as ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils.keras_utils import shifted_softplus
from .transform import Transform


@serializable(package="bayesflow.networks.coupling_flow")
class AffineTransform(Transform):
    def __init__(self, clamp: bool | int | float | None = 3.0, **kwargs):
        super().__init__(**kwargs)
        match clamp:
            case True:
                self.clamp_factor = 3.0
            case False:
                self.clamp_factor = None
            case int() | float():
                self.clamp_factor = float(clamp)
            case None:
                self.clamp_factor = None
            case _:
                raise ValueError(f"Invalid value for 'clamp': {clamp}")

    @property
    def params_per_dim(self):
        return 2

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        scale, shift = ops.split(parameters, 2, axis=-1)

        return {"scale": scale, "shift": shift}

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        scale = parameters["scale"]

        # constrain to positive values
        scale = shifted_softplus(scale)

        # soft clamp
        if self.clamp_factor is not None:
            scale = self.clamp_factor * keras.ops.tanh(scale)

        parameters["scale"] = scale
        return parameters

    def _forward(self, x: Tensor, parameters: dict[str, Tensor] = None) -> (Tensor, Tensor):
        z = parameters["scale"] * x + parameters["shift"]
        log_det = ops.sum(ops.log(parameters["scale"]), axis=-1)

        return z, log_det

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor] = None) -> (Tensor, Tensor):
        x = (z - parameters["shift"]) / parameters["scale"]
        log_det = -ops.sum(ops.log(parameters["scale"]), axis=-1)

        return x, log_det
