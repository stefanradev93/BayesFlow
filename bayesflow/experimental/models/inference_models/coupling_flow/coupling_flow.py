
import keras

from typing import Self, Sequence

from bayesflow.experimental.types import Tensor

from .coupling import Coupling


class CouplingFlow(keras.Model):
    def __init__(self, couplings: Sequence[Coupling], latent_distribution):
        super().__init__()

        self.couplings = list(couplings)
        self.latent_distribution = latent_distribution

    @classmethod
    def from_coupling(cls, coupling: Coupling, depth: int) -> Self:
        # TODO: add some convenience construction methods with clear names
        ...

    def forward(self, x: Tensor) -> Tensor:
        z = x
        for coupling in self.couplings:
            z = coupling.forward(z)

        return z

    def inverse(self, z: Tensor) -> Tensor:
        x = z
        for coupling in reversed(self.couplings):
            x = coupling.inverse(x)

        return x

    def forward_jacobian(self, x: Tensor) -> (Tensor, Tensor):
        z = x
        logdet = keras.ops.zeros(keras.ops.shape(x)[0])
        for coupling in self.couplings:
            z, det = coupling.forward_jacobian(z)
            logdet += det

        return z, logdet

    def inverse_jacobian(self, z: Tensor) -> (Tensor, Tensor):
        x = z
        logdet = keras.ops.zeros(keras.ops.shape(x)[0])
        for coupling in reversed(self.couplings):
            x, det = coupling.inverse_jacobian(x)
            logdet += det

        return x, logdet

    def call(self, x):
        return self.forward_jacobian(x)

    def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):
        z, logdet = y_pred
        log_prob = self.latent_distribution.log_prob(z)

        # change of variables
        nll = -(log_prob + logdet)

        return keras.ops.mean(nll, axis=0)
