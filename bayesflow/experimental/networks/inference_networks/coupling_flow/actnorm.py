
import keras
from keras import ops as K

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

    def call(self, target: Tensor, inverse=False) -> Tensor:
        """Performs one pass through the activation normalization layer (either inverse or forward) and normalizes
        the last axis of `target`.

        Parameters
        ----------
        target     : keras.Tensor of shape (batch_size, ...)
            the target variables of interest, i.e., parameters for posterior estimation
        inverse    : bool, optional, default: False
            Flag indicating whether to run the block forward or backwards

        Returns
        -------
        (z, log_det_J)          :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (,)
        (target, log_det_J)     :  tf.Tensor
            If inverse=True: The inversely transformed targets, shape == target.shape
        """

        if not inverse:
            return self.forward(target)
        else:
            return self.inverse(target)

    def forward(self, x: Tensor) -> Tensor:
        z = self.scale * x + self.bias
        logdet = K.sum(K.log(K.abs(self.scale)), axis=-1)
        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        x = (z - self.bias) / self.scale
        logdet = -K.sum(K.log(K.abs(self.scale)), axis=-1)
        return x, logdet
