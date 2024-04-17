
from typing import Self, Sequence

import keras

from bayesflow.experimental.simulation import Distribution, find_distribution
from bayesflow.experimental.types import Shape, Tensor

from .couplings import DualCoupling
from .transforms import find_transform


class CouplingFlow(keras.Sequential):
    """ Implements a coupling flow as a sequence of dual couplings with swap permutations """
    def __init__(self, couplings: Sequence[DualCoupling], base_distribution: Distribution):
        super().__init__(couplings)
        self.base_distribution = base_distribution

    @classmethod
    def uniform(
            cls,
            subnet_constructor: callable,
            features: int,
            conditions: int,
            layers: int,
            transform="affine",
            base_distribution="normal",
    ) -> Self:
        """ Construct a uniform coupling flow, consisting of dual couplings with a single type of transform. """
        transform = find_transform(transform)
        base_distribution = find_distribution(base_distribution, shape=(features,))

        couplings = []
        for _ in range(layers):
            c = DualCoupling(subnet_constructor, features, conditions, transform)
            couplings.append(c)

        return cls(couplings, base_distribution)

    def call(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):
        z, logdet = y_pred
        log_prob = self.base_distribution.log_prob(z)
        nll = -keras.ops.mean(log_prob + logdet, axis=0)

        return nll

    def compute_metrics(self, x, y, y_pred, **kwargs):
        return {}

    def forward(self, x, c=None):
        z = x
        logdet = keras.ops.zeros(keras.ops.shape(x)[0])
        for coupling in self.layers:
            z, det = coupling.forward(z, c)
            logdet += det

        return z, logdet

    def inverse(self, z, c=None):
        x = z
        logdet = keras.ops.zeros(keras.ops.shape(x)[0])
        for coupling in reversed(self.layers):
            x, det = coupling.inverse(x, c)
            logdet += det

        return x, logdet

    def sample(self, batch_shape: Shape):
        z = self.base_distribution.sample(batch_shape)
        x, _ = self.inverse(z)

        return x

    def log_prob(self, x: Tensor) -> Tensor:
        z, logdet = self.forward(x)
        log_prob = self.base_distribution.log_prob(z)

        return log_prob + logdet
