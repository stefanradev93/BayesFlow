
import keras
from keras import layers, regularizers
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Tensor


@register_keras_serializable(package="bayesflow.networks")
class MultiHeadAttentionBlock(keras.Layer):
    """Implements the MAB block from [1] which represents learnable cross-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        input_dim: int,
        key_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.05,
        num_dense_fc: int = 2,
        dense_units: int = 128,
        dense_activation: str | callable = "gelu",
        kernel_regularizer: regularizers.Regularizer | None = None,
        kernel_initializer: str = "he_uniform",
        bias_regularizer: regularizers.Regularizer | None = None,
        use_bias=True,
        layer_norm: bool = True,
        **kwargs
    ):
        """Creates a multi-head attention block which will typically be used as part of a
        set transformer architecture according to [1]. Corresponds to standard cross-attention.

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

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ln_pre = layers.LayerNormalization() if layer_norm else None
        self.fc = keras.Sequential()
        for _ in range(num_dense_fc):
            self.fc.add(layers.Dense(
                units=dense_units,
                activation=dense_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                bias_regularizer=bias_regularizer,
                use_bias=use_bias
            ))
            self.fc.add(layers.Dropout(dropout))
        self.fc.add(layers.Dense(input_dim))
        self.ln_post = layers.LayerNormalization() if layer_norm else None

    def call(self, x: Tensor, y: Tensor, **kwargs):
        """Performs the forward pass through the attention layer.

        Parameters
        ----------
        x : Tensor
            Input of shape (batch_size, set_size_x, input_dim)
        y : Tensor
            Input of shape (batch_size, set_size_y, input_dim)

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size_x, input_dim)
        """

        h = x + self.att(x, y, y, **kwargs)
        if self.ln_pre is not None:
            h = self.ln_pre(h, **kwargs)
        out = h + self.fc(h, **kwargs)
        if self.ln_post is not None:
            out = self.ln_post(out, **kwargs)
        return out
