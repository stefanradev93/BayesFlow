
import keras
from keras import Sequential
from keras.api.layers import Dense

from .invariant_module import InvariantModule


class EquivariantModule(keras.Model):
    """Implements an equivariant module performing an equivariant transform.

    For details and justification, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an equivariant module according to [1] which combines equivariant transforms
        with nested invariant transforms, thereby enabling interactions between set members.

        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model`` constructor.
        """

        super().__init__(**kwargs)

        self.invariant_module = InvariantModule(settings)
        self.s3 = Sequential([Dense(**settings["dense_s3_args"]) for _ in range(settings["num_dense_s3"])])

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable equivariant transform.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, ..., x_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, ..., equiv_dim)
        """

        # Store shape of x, will be (batch_size, ..., some_dim)
        shape = keras.ops.shape(x)

        # Example: Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module(x, **kwargs)
        out_inv = keras.ops.expand_dims(out_inv, -2)
        tiler = [1] * len(shape)
        tiler[-2] = shape[-2]
        out_inv_rep = keras.ops.tile(out_inv, tiler)

        # Concatenate each x with the repeated invariant embedding
        out_c = keras.ops.concatenate([x, out_inv_rep], axis=-1)

        # Pass through equivariant func
        out = self.s3(out_c, **kwargs)
        return out
