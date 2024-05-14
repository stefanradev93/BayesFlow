
import math

import keras

from bayesflow.experimental.types import Tensor
from .coupling import Coupling
from ..transforms import Transform


class DualCoupling(keras.Layer):
    def __init__(
        self,
        subnet: keras.Model | keras.layers.Layer,
        target_dim: int,
        transform: Transform
    ):
        super().__init__()

        self.coupling1 = Coupling(
            subnet=subnet,
            target_dim=math.floor(target_dim / 2),
            transform=transform,
        )
        self.coupling2 = Coupling(
            subnet=subnet,
            target_dim=math.ceil(target_dim / 2),
            transform=transform,
        )

    def call(self, x: Tensor, c=None, forward=True, **kwargs) -> (Tensor, Tensor):
        if forward:
            self.forward(x, c, **kwargs)
        self.inverse(x, c)

    def forward(self, x: Tensor, c=None, **kwargs) -> (Tensor, Tensor):
        z, det1 = self.coupling1.forward(x, c, **kwargs)
        z, det2 = self.coupling2.forward(z, c, **kwargs)

        return z, det1 + det2

    def inverse(self, z: Tensor, c=None) -> (Tensor, Tensor):
        x, det2 = self.coupling2.inverse(z, c)
        x, det1 = self.coupling1.inverse(x, c)

        return x, det1 + det2
