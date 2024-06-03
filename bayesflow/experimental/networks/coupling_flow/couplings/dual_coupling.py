
import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.experimental.types import Tensor
from .single_coupling import SingleCoupling
from ..invertible_layer import InvertibleLayer


@register_keras_serializable(package="bayesflow.networks.coupling_flow")
class DualCoupling(InvertibleLayer):
    def __init__(self, subnet: str = "resnet", transform: str = "affine", **kwargs):
        super().__init__(**kwargs)
        self.coupling1 = SingleCoupling(subnet, transform, name=f"CouplingA")
        self.coupling2 = SingleCoupling(subnet, transform, name=f"CouplingB")
        self.pivot = None

    def build(self, input_shape):
        self.pivot = input_shape[-1] // 2

    def call(self, xz: Tensor, conditions: Tensor = None, inverse: bool = False, **kwargs) -> (Tensor, Tensor):
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

    def _forward(self, x: Tensor, conditions: Tensor = None, **kwargs) -> (Tensor, Tensor):
        """ Transform (x1, x2) -> (g(x1; f(x2; x1)), f(x2; x1)) """
        x1, x2 = x[..., :self.pivot], x[..., self.pivot:]
        (z1, z2), log_det1 = self.coupling1(x1, x2, conditions=conditions, **kwargs)
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, **kwargs)

        z = keras.ops.concatenate([z1, z2], axis=-1)
        log_det = log_det1 + log_det2

        return z, log_det

    def _inverse(self, z: Tensor, conditions: Tensor = None, **kwargs) -> (Tensor, Tensor):
        """ Transform (g(x1; f(x2; x1)), f(x2; x1)) -> (x1, x2) """
        z1, z2 = z[..., :self.pivot], z[..., self.pivot:]
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, inverse=True, **kwargs)
        (x1, x2), log_det1 = self.coupling1(z1, z2, conditions=conditions, inverse=True, **kwargs)

        x = keras.ops.concatenate([x1, x2], axis=-1)
        log_det = log_det1 + log_det2

        return x, log_det
