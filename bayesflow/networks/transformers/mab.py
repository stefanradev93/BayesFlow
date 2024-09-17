import keras
from keras import layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor


@serializable(package="bayesflow.networks")
class MultiHeadAttentionBlock(keras.Layer):
    """Implements the MAB block from [1] which represents learnable cross-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        key_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.05,
        num_dense_feedforward: int = 2,
        output_dim: int = None,
        dense_units: int = 128,
        dense_activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        use_bias=True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """Creates a multi-head attention block which will typically be used as part of a
        set transformer architecture according to [1]. Corresponds to standard cross-attention.

        Parameters
        ----------
        ##TODO
        """

        super().__init__(**kwargs)

        if output_dim is None:
            output_dim = key_dim
        self.projector = layers.Dense(output_dim)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            output_shape=output_dim,
            use_bias=use_bias,
        )
        self.ln_pre = layers.LayerNormalization() if layer_norm else None
        self.feedforward = keras.Sequential()
        for _ in range(num_dense_feedforward):
            self.feedforward.add(
                layers.Dense(
                    units=dense_units,
                    activation=dense_activation,
                    kernel_initializer=kernel_initializer,
                    use_bias=use_bias,
                )
            )
            self.feedforward.add(layers.Dropout(dropout))
        self.feedforward.add(layers.Dense(output_dim))
        self.ln_post = layers.LayerNormalization() if layer_norm else None

    def call(self, set_x: Tensor, set_y: Tensor, **kwargs) -> Tensor:
        """Performs the forward pass through the attention layer.

        Parameters
        ----------
        set_x : Tensor
            Input of shape (batch_size, set_size_x, input_dim), which will
            play the role of a query (Q).
        set_y : Tensor
            Input of shape (batch_size, set_size_y, input_dim), which will
            play the role of key (K) and value (V).

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size_x, input_dim)
        """

        training = kwargs.get("training", False)
        h = self.projector(set_x) + self.att(set_x, set_y, set_y, **kwargs)
        if self.ln_pre is not None:
            h = self.ln_pre(h, training=training)
        out = h + self.feedforward(h, training=training)
        if self.ln_post is not None:
            out = self.ln_post(out, training=training)
        return out

    def build(self, input_shape):
        super().build(input_shape)
