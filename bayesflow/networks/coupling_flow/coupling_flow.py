
from typing import Tuple, Union

import keras
from keras.saving import register_keras_serializable

from bayesflow.types import Tensor
from bayesflow.utils import find_permutation, keras_kwargs

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
        permutation: str | None = None,
        use_actnorm: bool = True,
        base_distribution: str = "normal",
        **kwargs
    ):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))

        self.depth = depth

        self.invertible_layers = []
        for i in range(depth):
            if use_actnorm:
                self.invertible_layers.append(ActNorm(**kwargs))

            self.invertible_layers.append(DualCoupling(subnet, transform, **kwargs))

            if (p := find_permutation(permutation, **kwargs)) is not None:
                self.invertible_layers.append(p)

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)

        xz = keras.ops.zeros(xz_shape)
        if conditions_shape is None:
            conditions = None
        else:
            conditions = keras.ops.zeros(conditions_shape)

        # build nested layers with forward pass
        self.call(xz, conditions=conditions)

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        inverse: bool = False,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

    def _forward(
        self, x: Tensor,
        conditions: Tensor = None,
        jacobian: bool = False,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        z = x
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1])
        for layer in self.invertible_layers:
            z, det = layer(z, conditions=conditions, inverse=False, **kwargs)
            log_det += det

        if jacobian:
            return z, log_det
        return z

    def _inverse(
        self, z: Tensor,
        conditions: Tensor = None,
        jacobian: bool = False,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        x = z
        log_det = keras.ops.zeros(keras.ops.shape(z)[:-1])
        for layer in reversed(self.invertible_layers):
            x, det = layer(x, conditions=conditions, inverse=True, **kwargs)
            log_det += det

        if jacobian:
            return x, log_det
        return x

    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        inference_variables = data["inference_variables"]
        inference_conditions = data.get("inference_conditions")

        z, log_det = self(inference_variables, conditions=inference_conditions, inverse=False, jacobian=True)
        log_prob = self.base_distribution.log_prob(z)
        loss = -keras.ops.mean(log_prob + log_det)

        return {"loss": loss}
