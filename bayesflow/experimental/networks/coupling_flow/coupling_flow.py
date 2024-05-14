
from typing import Sequence

import keras

from bayesflow.experimental.simulation import Distribution, find_distribution
from bayesflow.experimental.types import Shape, Tensor
from .couplings import AllInOneCoupling


class CouplingFlow(keras.Sequential):
    """ Implements a coupling flow as a sequence of dual couplings with swap permutations """
    def __init__(self, couplings: Sequence[AllInOneCoupling], base_distribution: Distribution):
        super().__init__(couplings)
        self.base_distribution = base_distribution

    @classmethod
    def all_in_one(
            cls,
            target_dim: int,
            num_layers: int,
            subnet_builder="default",
            transform="affine",
            permutation="fixed",
            act_norm=True,
            base_distribution="normal",
            **kwargs
    ) -> "CouplingFlow":
        """ Construct a uniform coupling flow, consisting of dual couplings with a single type of transform. """

        base_distribution = find_distribution(base_distribution, shape=(target_dim,))

        couplings = []
        for _ in range(num_layers):
            layer = AllInOneCoupling(subnet_builder, target_dim, transform, permutation, act_norm, **kwargs)
            couplings.append(layer)

        return cls(couplings, base_distribution)

    def call(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):
        z, log_det = y_pred
        log_prob = self.base_distribution.log_prob(z)
        nll = -keras.ops.mean(log_prob + log_det)

        return nll

    def compute_metrics(self, x, y, y_pred, **kwargs):
        return {}

    def forward(self, targets, conditions=None, **kwargs) -> (Tensor, Tensor):
        latents = targets
        log_det = 0.
        for coupling in self.layers:
            latents, det = coupling.forward(latents, conditions, **kwargs)
            log_det += det

        return latents, log_det

    def inverse(self, latents, conditions=None) -> (Tensor, Tensor):
        targets = latents
        log_det = 0.
        for coupling in reversed(self.layers):
            targets, det = coupling.inverse(targets, conditions)
            log_det += det

        return targets, log_det

    def sample(self, batch_shape: Shape, conditions=None) -> Tensor:
        latents = self.base_distribution.sample(batch_shape)
        targets, _ = self.inverse(latents, conditions)

        return targets

    def log_prob(self, targets: Tensor, conditions=None, **kwargs) -> Tensor:
        latents, log_det = self.forward(targets, conditions, **kwargs)
        log_prob = self.base_distribution.log_prob(latents)

        return log_prob + log_det
