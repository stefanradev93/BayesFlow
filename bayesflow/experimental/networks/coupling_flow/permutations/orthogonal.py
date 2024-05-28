from keras import ops

from bayesflow.experimental.types import Shape, Tensor
from ..invertible_layer import InvertibleLayer


class OrthogonalPermutation(InvertibleLayer):
    """Implements a learnable orthogonal transformation according to [1]. Can be
    used as an alternative to a fixed ``Permutation`` layer.

    [1] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1
    convolutions. Advances in neural information processing systems, 31.
    """

    def __init__(self, **kwargs):
        # TODO: saving and loading
        super().__init__(**kwargs)
        self.weight = None

    def build(self, input_shape: Shape) -> None:
        self.weight = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="orthogonal",
            trainable=True
        )

    def _forward(self, x: Tensor) -> (Tensor, Tensor):
        z = ops.matmul(x, self.weight)
        log_det = ops.log(ops.abs(ops.det(self.weight)))

        if ops.ndim(x) > 2:
            log_det = ops.multiply(log_det, ops.shape(x)[1])

        return z, log_det

    def _inverse(self, z: Tensor) -> (Tensor, Tensor):
        weight = ops.inv(self.weight)

        x = ops.matmul(z, weight)
        log_det = ops.log(ops.abs(ops.det(weight)))

        if ops.ndim(z) > 2:
            log_det = ops.multiply(log_det, ops.shape(z)[1])

        return x, log_det
