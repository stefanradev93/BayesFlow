
from typing import Tuple, Union

import keras
from keras.saving import register_keras_serializable

from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import keras_kwargs
from .actnorm import ActNorm
from .couplings import DualCoupling
from .permutations import OrthogonalPermutation, RandomPermutation, Swap
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
        permutation: str = "random",
        use_actnorm: bool = True,
        **kwargs
    ):
        """TODO"""

        super().__init__(**keras_kwargs(kwargs))

        self._layers = []
        for i in range(depth):
            if use_actnorm:
                self._layers.append(ActNorm())
            self._layers.append(DualCoupling(subnet, transform, **kwargs))
            if permutation.lower() == "random":
                self._layers.append(RandomPermutation())
            elif permutation.lower() == "swap":
                self._layers.append(Swap())
            elif permutation.lower() == "learnable":
                self._layers.append(OrthogonalPermutation())

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)
        if conditions_shape is None:
            self.call(keras.KerasTensor(xz_shape))
        else:
            self.call(keras.KerasTensor(xz_shape), conditions=keras.KerasTensor(conditions_shape))

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        inverse: bool = False, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

    def _forward(
        self,
        x: Tensor,
        conditions: Tensor = None,
        jacobian: bool = False,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        z = x
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1])
        for layer in self._layers:
            if isinstance(layer, DualCoupling):
                z, det = layer(z, conditions=conditions, inverse=False, **kwargs)
            else:
                z, det = layer(z, inverse=False, **kwargs)
            log_det += det

        if jacobian:
            return z, log_det
        return z

    def _inverse(self, z: Tensor, conditions: Tensor = None, jacobian: bool = False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = z
        log_det = keras.ops.zeros(keras.ops.shape(z)[:-1])
        for layer in reversed(self._layers):
            if isinstance(layer, DualCoupling):
                x, det = layer(x, conditions=conditions, inverse=True, **kwargs)
            else:
                x, det = layer(x,  inverse=True, **kwargs)
            log_det += det

        if jacobian:
            return x, log_det
        return x

    def compute_loss(self, x: Tensor = None, conditions: Tensor = None, **kwargs):
        z, log_det = self(x, conditions=conditions, inverse=False, jacobian=True, **kwargs)
        log_prob = self.base_distribution.log_prob(z)
        nll = -keras.ops.mean(log_prob + log_det)

        return nll
