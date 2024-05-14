
import keras
from keras import ops

from bayesflow.experimental.types import Tensor


class ActNorm(keras.Layer):
    """Implements an Activation Normalization (ActNorm) Layer.
    Activation Normalization is learned invertible normalization, using
    a Scale (s) and Bias (b) vector::

       y = s * x + b (forward)
       x = (y - b) / s (inverse)

    References
    ----------

    .. [1] Kingma, Diederik P., and Prafulla Dhariwal.
       "Glow: Generative flow with invertible 1x1 convolutions."
       arXiv preprint arXiv:1807.03039 (2018).

    .. [2] Salimans, Tim, and Durk P. Kingma.
       "Weight normalization: A simple reparameterization to accelerate
       training of deep neural networks."
       Advances in neural information processing systems 29 (2016): 901-909.
    """

    def __init__(self, target_dim: int, **kwargs):
        """Creates an instance of an ActNorm Layer as proposed by [1].

        Parameters
        ----------
        target_dim            : int
            The dimensionality of the target (e.g., parameter) space
        """

        super().__init__(**kwargs)

        self.scale = self.add_weight(
            shape=(target_dim, ),
            initializer='ones',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(target_dim, ),
            initializer='zeros',
            trainable=True
        )

    def call(self, target: Tensor, forward=True) -> Tensor:
        """Performs one pass through the activation normalization layer (either inverse or forward) and normalizes
        the last axis of `target`.

        Parameters
        ----------
        target     : keras.Tensor of shape (batch_size, ...)
            the target variables of interest, i.e., parameters for posterior estimation
        forward    : bool, optional, default: False
            Controls if the current pass is forward (``forward=True``) or inverse (``forward=False``).

        Returns
        -------
        (z, log_det_J)           :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (,)
        (target, -log_det_J)     :  tf.Tensor
            If inverse=True: The inversely transformed targets, shape == target.shape
        """

        if forward:
            return self.forward(target)
        return self.inverse(target)

    def forward(self, x: Tensor) -> Tensor:
        z = self.scale * x + self.bias
        log_det = ops.sum(ops.log(ops.abs(self.scale)), axis=-1)
        return z, log_det

    def inverse(self, z: Tensor) -> Tensor:
        x = (z - self.bias) / self.scale
        log_det = -ops.sum(ops.log(ops.abs(self.scale)), axis=-1)
        return x, log_det
