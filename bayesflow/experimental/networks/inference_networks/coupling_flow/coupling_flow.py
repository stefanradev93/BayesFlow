
from typing import Sequence

import keras

from bayesflow.experimental.simulation import Distribution, find_distribution
from bayesflow.experimental.types import Shape, Tensor

from .couplings import AllInOneCoupling
from .transforms import find_transform


class CouplingFlow(keras.Sequential):
    """ Implements a coupling flow as a sequence of dual couplings with swap permutations """
    def __init__(self, couplings: Sequence[AllInOneCoupling], base_distribution: Distribution):
        super().__init__(couplings)
        self.base_distribution = base_distribution

    @classmethod
    def uniform(
            cls,
            subnet_constructor: callable,
            target_dim: int,
            num_layers: int,
            transform="affine",
            permutation='fixed',
            act_norm=True,
            base_distribution="normal",
    ) -> "CouplingFlow":
        """ Construct a uniform coupling flow, consisting of dual couplings with a single type of transform. """

        transform = find_transform(transform)
        base_distribution = find_distribution(base_distribution, shape=(target_dim,))

        couplings = []
        for _ in range(num_layers):
            layer = AllInOneCoupling(subnet_constructor, target_dim, transform, permutation, act_norm)
            couplings.append(layer)

        return cls(couplings, base_distribution)

    def call(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):
        z, log_det = y_pred
        log_prob = self.base_distribution.log_prob(z)
        nll = -keras.ops.mean(log_prob + log_det, axis=0)

        return nll

    def compute_metrics(self, x, y, y_pred, **kwargs):
        return {}

    def forward(self, x, c=None):
        z = x
        log_det = 0.
        for coupling in self.layers:
            z, det = coupling.forward(z, c)
            log_det += det

        return z, log_det

    def inverse(self, z, c=None):
        x = z
        log_det = 0.
        for coupling in reversed(self.layers):
            x, det = coupling.inverse(x, c)
            log_det += det

        return x, log_det

    def sample(self, batch_shape: Shape):
        z = self.base_distribution.sample(batch_shape)
        x, _ = self.inverse(z)

        return x

    def log_prob(self, x: Tensor, **kwargs) -> Tensor:
        z, log_det = self.forward(x, **kwargs)
        log_prob = self.base_distribution.log_prob(z)

        return log_prob + log_det
