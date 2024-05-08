
import keras
from keras import ops

from bayesflow.experimental.types import Tensor


class FixedPermutation(keras.layers.Layer):
    def __init__(self, indices: Tensor):
        super().__init__()
        self.indices = indices
        self.inverse_indices = keras.ops.argsort(indices, axis=0)

    def forward(self, x: Tensor) -> Tensor:
        return keras.ops.take(x, self.indices, axis=-1)

    def inverse(self, z: Tensor) -> Tensor:
        return keras.ops.take(z, self.inverse_indices, axis=-1)

    @classmethod
    def swap(cls, target_dim: int):
        indices = keras.ops.arange(0, target_dim)
        indices = keras.ops.roll(indices, shift=target_dim // 2, axis=0)
        return cls(indices)

    @classmethod
    def random(cls, target_dim: int):
        indices = keras.ops.arange(0, target_dim)
        indices = keras.random.shuffle(indices)
        return cls(indices)


class LearnablePermutation(keras.layers.Layer):
    """Implements a learnable orthogonal transformation according to [1]. Can be
    used as an alternative to a fixed ``Permutation`` layer.

    [1] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1
    convolutions. Advances in neural information processing systems, 31.
    """

    def __init__(self, input_dim):
        """Creates an invertible orthogonal transformation (generalized permutation)

        Parameters
        ----------
        input_dim  : int
            Ihe dimensionality of the input to the (conditional) coupling layer.
        """

        super().__init__()

        self.W = self.add_weight(
            shape=(input_dim, input_dim),
            initializer='orthogonal',
            trainable=True
        )

    def call(self, target: Tensor, inverse=False) -> Tensor:
        """Transforms a batch of target vectors over the last axis through an approximately
        orthogonal transform.

        Parameters
        ----------
        target   : tf.Tensor of shape (batch_size, ...)
            The target vector to be rotated over its last axis.
        inverse  : bool, optional, default: False
            Controls if the current pass is forward (``inverse=False``) or inverse (``inverse=True``).

        Returns
        -------
        out      : tf.Tensor of the same shape as `target`.
            The (un-)rotated target vector.
        """

        if not inverse:
            return self.forward(target)
        return self.inverse(target)

    def forward(self, target: Tensor) -> Tensor:
        shape = ops.shape(target)
        z = ops.matmul(target, self.W)
        log_det = ops.log(ops.abs(ops.det(self.W)))
        if len(shape) > 2:
            log_det = ops.multiply(log_det, shape[1])
        return z, log_det

    def inverse(self, z: Tensor) -> Tensor:
        shape = ops.shape(z)
        W_inv = ops.inv(self.W)
        target = ops.matmul(z, W_inv)
        log_det = -ops.log(ops.abs(ops.det(self.W)))
        if len(shape) > 2:
            log_det = ops.multiply(log_det, shape[1])
        return target, log_det
