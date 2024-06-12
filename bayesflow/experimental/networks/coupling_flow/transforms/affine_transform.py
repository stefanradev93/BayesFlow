
import keras.ops as ops
from keras.saving import register_keras_serializable

from bayesflow.experimental.types import Tensor
from .transform import Transform


@register_keras_serializable(package="bayesflow.networks.coupling_flow")
class AffineTransform(Transform):
    def __init__(self, clamp_factor: int | float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.clamp_factor = clamp_factor

    @property
    def params_per_dim(self):
        return 2

    @classmethod
    def from_config(cls, config):
        clamp_factor = config.pop("clamp_factor")

        return cls(clamp_factor, **config)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "clamp_factor": self.clamp_factor,
        }

        return base_config | config

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        scale, shift = ops.split(parameters, 2, axis=-1)

        return {"scale": scale, "shift": shift}

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        s = parameters["scale"]
        parameters["scale"] = 1 / (1 + ops.exp(-s)) * ops.sqrt(1 + ops.abs(s + self.clamp_factor))

        return parameters

    def _forward(self, x: Tensor, parameters: dict[str, Tensor] = None) -> (Tensor, Tensor):
        z = parameters["scale"] * x + parameters["shift"]
        log_det = ops.sum(ops.log(parameters["scale"]), axis=-1)

        return z, log_det

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor] = None) -> (Tensor, Tensor):
        x = (z - parameters["shift"]) / parameters["scale"]
        log_det = -ops.sum(ops.log(parameters["scale"]), axis=-1)

        return x, log_det
