
from typing import Tuple, Union

import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.experimental.types import Tensor
from .actnorm import ActNorm
from .couplings import DualCoupling
from ..inference_network import InferenceNetwork


@register_keras_serializable(package="bayesflow.networks")
class CouplingFlow(InferenceNetwork):
    """ Implements a coupling flow as a sequence of dual couplings with permutations and activation
    normalization. Incorporates ideas from [1-5].

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
    def __init__(
            self,
            depth: int = 6,
            subnet: str = "resnet",
            transform: str = "affine",
            use_actnorm: bool = True, **kwargs
    ):
        super().__init__(**kwargs)

        self._layers = []
        for _ in range(depth):
            if use_actnorm:
                self._layers.append(ActNorm())
            self._layers.append(DualCoupling(subnet, transform))

    def build(self, input_shape):
        super().build(input_shape)
        self.call(keras.KerasTensor(input_shape))

    def call(self, xz: Tensor, inverse: bool = False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if inverse:
            return self._inverse(xz, **kwargs)
        return self._forward(xz, **kwargs)

    def _forward(self, x: Tensor, jacobian: bool = False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        z = x
        log_det = 0.0
        for layer in self._layers:
            z, det = layer(z, inverse=False, **kwargs)
            log_det += det

        if jacobian:
            return z, log_det
        return z

    def _inverse(self, z: Tensor, jacobian: bool = False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = z
        log_det = 0.0
        for layer in reversed(self._layers):
            x, det = layer(x, inverse=True, **kwargs)
            log_det += det

        if jacobian:
            return x, log_det
        return x

    def compute_loss(self, x: Tensor = None, **kwargs):
        z, log_det = self(x, inverse=False, jacobian=True, **kwargs)
        log_prob = self.base_distribution.log_prob(z)
        nll = -keras.ops.mean(log_prob + log_det)

        return nll
