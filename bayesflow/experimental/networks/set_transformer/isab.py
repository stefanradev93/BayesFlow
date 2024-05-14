
import keras
import keras.ops as ops

from bayesflow.experimental.types import Tensor
from .mab import MultiHeadAttentionBlock


class InducedSetAttentionBlock(keras.Layer):
    """Implements the ISAB block from [1] which represents learnable self-attention specifically
    designed to deal with large sets via a learnable set of "inducing points".

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        input_dim,
        attention_settings,
        num_dense_fc,
        dense_settings,
        layer_norm,
        num_inducing_points,
        **kwargs
    ):
        """Creates a self-attention attention block with inducing points (ISAB) which will typically
        be used as part of a set transformer architecture according to [1].

        Parameters
        ----------
        input_dim           : int
            The dimensionality of the input data (last axis).
        attention_settings  : dict
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention.
        num_dense_fc        : int
            The number of hidden layers for the internal feedforward network
        dense_settings      : dict
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer
        layer_norm          : boolean
            Whether layer normalization before and after attention + feedforward
        num_inducing_points : int
            The number of inducing points. Should be lower than the smallest set size
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        self.inducing_points = self.add_weight(
            shape=(num_inducing_points, input_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.mab0 = MultiHeadAttentionBlock(input_dim, attention_settings, num_dense_fc, dense_settings, layer_norm)
        self.mab1 = MultiHeadAttentionBlock(input_dim, attention_settings, num_dense_fc, dense_settings, layer_norm)

    def call(self, x: Tensor, **kwargs):
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        x   : Tensor
            Input of shape (batch_size, set_size, input_dim)

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        batch_size = ops.shape(x)[0]
        inducing_points_expanded = ops.expand_dims(self.inducing_points, axis=0)
        inducing_points_tiled = ops.tile(inducing_points_expanded, [batch_size, 1, 1])
        h = self.mab0(inducing_points_tiled, x, **kwargs)
        return self.mab1(x, h, **kwargs)
