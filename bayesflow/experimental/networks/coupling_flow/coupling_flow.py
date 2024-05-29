
from typing import Sequence

import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Shape, Tensor
from .actnorm import ActNorm
from .couplings import DualCoupling
from .invertible_layer import InvertibleLayer


@register_keras_serializable(package="bayesflow.networks")
class CouplingFlow(keras.Model, InvertibleLayer):
    """ Implements a coupling flow as a sequence of dual couplings with permutations and activation
    normalization. Incorporates ideas from [1-4].

    [1] Kingma, D. P., & Dhariwal, P. (2018).
    Glow: Generative flow with invertible 1x1 convolutions.
    Advances in Neural Information Processing Systems, 31.

    [2] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
    Neural spline flows. Advances in Neural Information Processing Systems, 32.

    [3] Ardizzone, L., Kruse, J., Lüth, C., Bracher, N., Rother, C., & Köthe, U. (2020).
    Conditional invertible neural networks for diverse image-to-image translation.
    In DAGM German Conference on Pattern Recognition (pp. 373-387). Springer, Cham.

    [4] Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020).
    BayesFlow: Learning complex stochastic models with invertible neural networks.
    IEEE Transactions on Neural Networks and Learning Systems.

    [5] Alexanderson, S., & Henter, G. E. (2020).
    Robust model training and generalisation with Studentising flows.
    arXiv preprint arXiv:2006.06599.
    """
    def __init__(self, invertible_layers: Sequence[InvertibleLayer], base_distribution: str = "normal", **kwargs):
        super().__init__(**kwargs)
        self.invertible_layers = list(invertible_layers)
        self.base_distribution = base_distribution

        # register variables
        for layer in self.invertible_layers:
            for variable in layer.variables:
                self._track_variable(variable)

    @classmethod
    def new(
        cls,
        depth: int = 6,
        subnet: str = "resnet",
        transform: str = "affine",
        base_distribution: str = "normal",
        use_actnorm: bool = True,
        **kwargs
    ):

        layers = []
        for i in range(depth):
            if use_actnorm:
                layers.append(ActNorm())
            layers.append(DualCoupling.new(subnet, transform))

        return cls(layers, base_distribution, **kwargs)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        couplings = deserialize_keras_object(config.pop("invertible_layers"))
        base_distribution = config.pop("base_distribution")

        return cls(couplings, base_distribution, **config)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "invertible_layers": serialize_keras_object(self.invertible_layers),
            "base_distribution": self.base_distribution,
        }

        return base_config | config

    def build(self, input_shape):
        # nothing to do here, since we do not know the conditions yet
        pass

    def call(self, x: Tensor, conditions: any = None, inverse: bool = False) -> (Tensor, Tensor):
        if inverse:
            return self._inverse(x, conditions)
        return self._forward(x, conditions)

    def _forward(self, x: Tensor, conditions: any = None) -> (Tensor, Tensor):
        z = x
        log_det = 0.0
        for layer in self.invertible_layers:
            z, det = layer(z, conditions=conditions)
            log_det += det

        return z, log_det

    def _inverse(self, z: Tensor, conditions: any = None) -> (Tensor, Tensor):
        x = z
        log_det = 0.0
        for layer in reversed(self.invertible_layers):
            x, det = layer(x, conditions=conditions, inverse=True)
            log_det += det

        return x, log_det

    def sample(self, batch_shape: Shape, conditions=None) -> Tensor:
        z = self.base_distribution.sample(batch_shape)
        x, _ = self(z, conditions, inverse=True)

        return x

    def log_prob(self, x: Tensor, conditions=None) -> Tensor:
        z, log_det = self(x, conditions)
        log_prob = self.base_distribution.log_prob(z)

        return log_prob + log_det

    def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):
        z, log_det = y_pred
        log_prob = self.base_distribution.log_prob(z)
        nll = -keras.ops.mean(log_prob + log_det)

        return nll

    def compute_metrics(self, x, y, y_pred, **kwargs):
        return {}
