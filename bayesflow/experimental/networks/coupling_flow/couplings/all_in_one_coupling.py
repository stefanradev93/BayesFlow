
import keras

from .dual_coupling import DualCoupling
from ..actnorm import ActNorm
from ..permutations import FixedPermutation, OrthogonalPermutation
from ..transforms import Transform


class AllInOneCoupling(keras.Layer):
    """ Implements a single coupling layer, followed by a permutation. """
    def __init__(
        self,
        subnet: keras.Model | keras.layers.Layer,
        target_dim: int,
        transform: Transform,
        permutation: str,
        act_norm: bool,
    ):
        super().__init__()
        self.dual_coupling = DualCoupling(subnet, target_dim, transform)

        if permutation == "fixed":
            self.permutation = FixedPermutation.swap(target_dim)
        else:
            self.permutation = OrthogonalPermutation(target_dim)
        if act_norm:
            self.act_norm = ActNorm(target_dim)
        else:
            self.act_norm = None

    def call(self, x, c=None, forward=True, **kwargs):
        if forward:
            self.forward(x, c, **kwargs)
        self.inverse(x, c)

    def forward(self, x, c=None, **kwargs):
        """Performs a forward pass through the chain: optional activation normalization -> coupling -> permutation."""

        # Activation normalization step
        if self.act_norm is not None:
            x, log_det_a = self.act_norm.forward(x)
        else:
            log_det_a = 0.

        # Coupling transform step
        z, log_det_c = self.dual_coupling.forward(x, c, **kwargs)

        # Permutation step
        if isinstance(self.permutation, FixedPermutation):
            z = self.permutation.forward(z)
            log_det_p = 0.
        else:
            z, log_det_p = self.permutation.forward(z)
        log_det = log_det_a + log_det_c + log_det_p

        return z, log_det

    def inverse(self, z, c=None):
        """Performs an inverse pass through the chain: permutation -> coupling -> optional activation normalization."""

        # Permutation step
        if isinstance(self.permutation, FixedPermutation):
            z = self.permutation.inverse(z)
            log_det_p = 0.
        else:
            z, log_det_p = self.permutation.inverse(z)

        # Coupling transform step
        x, log_det_c = self.dual_coupling.inverse(z, c)

        # Activation normalization step
        if self.act_norm is not None:
            x, log_det_a = self.act_norm.inverse(x)
        else:
            log_det_a = 0.
        log_det = log_det_a + log_det_c + log_det_p

        return x, log_det
