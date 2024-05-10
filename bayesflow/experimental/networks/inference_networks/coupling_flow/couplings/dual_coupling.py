
import keras
import math

from .coupling import Coupling
from ..transforms import Transform


class DualCoupling(keras.Layer):
    def __init__(
        self,
        subnet_constructor: callable,
        target_dim: int,
        transform: Transform
    ):
        super().__init__()

        self.coupling1 = Coupling(
            subnet_constructor=subnet_constructor,
            target_dim=math.floor(target_dim / 2),
            transform=transform,
        )
        self.coupling2 = Coupling(
            subnet_constructor=subnet_constructor,
            target_dim=math.ceil(target_dim / 2),
            transform=transform,
        )

    def forward(self, x, c=None, **kwargs):
        z, det1 = self.coupling1.forward(x, c, **kwargs)
        z, det2 = self.coupling2.forward(z, c, **kwargs)

        return z, det1 + det2

    def inverse(self, z, c=None):
        x, det2 = self.coupling2.inverse(z, c)
        x, det1 = self.coupling1.inverse(x, c)

        return x, det1 + det2
