
import keras

from .mab import MultiHeadAttentionBlock


class SetAttentionBlock(keras.layers.Layer):
    """Implements the SAB block from [1] which represents learnable self-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        input_dim: int,
        attention_settings: dict,
        num_dense_fc: int,
        dense_settings: dict,
        layer_norm: bool,
        **kwargs
    ):
        """Creates a self-attention attention block which will typically be used as part of a
        set transformer architecture according to [1].

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
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        self.mab = MultiHeadAttentionBlock(
            input_dim, attention_settings, num_dense_fc, dense_settings, layer_norm)

    def call(self, x, **kwargs):
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

        return self.mab(x, x, **kwargs)
