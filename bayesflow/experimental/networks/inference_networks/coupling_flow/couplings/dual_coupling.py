
import keras

from .coupling import Coupling
from ..transforms import Transform


class DualCoupling(keras.Layer):
    """.Implements a dual coupling layer."""
    def __init__(
        self,
        subnet_constructor: callable,
        target_dim: int,
        transform: Transform
    ):
        super().__init__()

        coupling1_dim = target_dim // 2
        coupling2_dim = target_dim // 2 if target_dim % 2 == 0 else target_dim // 2 + 1

        self.coupling1 = Coupling(
            subnet_constructor=subnet_constructor,
            target_dim=coupling1_dim,
            transform=transform,
        )
        self.coupling2 = Coupling(
            subnet_constructor=subnet_constructor,
            target_dim=coupling2_dim,
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
