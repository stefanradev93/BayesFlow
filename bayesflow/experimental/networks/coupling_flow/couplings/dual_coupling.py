
import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.experimental.types import Tensor
from .single_coupling import SingleCoupling
from ..invertible_layer import InvertibleLayer


@register_keras_serializable(package="bayesflow.networks.coupling_flow")
class DualCoupling(InvertibleLayer):
    def __init__(self, subnet: str = "resnet", transform: str = "affine"):
        super().__init__()
        self.coupling1 = SingleCoupling(subnet, transform)
        self.coupling2 = SingleCoupling(subnet, transform)
        self.pivot = None

    def build(self, input_shape):
        self.pivot = input_shape[-1] // 2

    def call(
        self,
        xz: Tensor,
        conditions: any = None,
        inverse: bool = False,
        training: bool = False
    ) -> (Tensor, Tensor):

        if inverse:
            return self._inverse(xz, conditions=conditions)
        return self._forward(xz, conditions=conditions, training=training)

    def _forward(self, x: Tensor, conditions: any = None, training: bool = False) -> (Tensor, Tensor):
        """ Transform (x1, x2) -> (g(x1; f(x2; x1)), f(x2; x1)) """
        x1, x2 = x[..., :self.pivot], x[..., self.pivot:]
        (z1, z2), log_det1 = self.coupling1(x1, x2, conditions=conditions, training=training)
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, training=training)

        z = keras.ops.concatenate([z1, z2], axis=-1)
        log_det = log_det1 + log_det2

        return z, log_det

    def _inverse(self, z: Tensor, conditions: any = None) -> (Tensor, Tensor):
        """ Transform (g(x1; f(x2; x1)), f(x2; x1)) -> (x1, x2) """
        z1, z2 = z[..., :self.pivot], z[..., self.pivot:]
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, inverse=True)
        (x1, x2), log_det1 = self.coupling1(z1, z2, conditions=conditions, inverse=True)

        x = keras.ops.concatenate([x1, x2], axis=-1)
        log_det = log_det1 + log_det2

        return x, log_det
