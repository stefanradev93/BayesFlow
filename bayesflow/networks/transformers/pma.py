import keras
import keras.ops as ops
from keras import layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from .mab import MultiHeadAttentionBlock


@serializable(package="bayesflow.networks")
class PoolingByMultiHeadAttention(keras.Layer):
    """Implements the pooling with multi-head attention (PMA) block from [1] which represents
    a permutation-invariant encoder for set-based inputs.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.

    Note: Currently works only on 3D inputs but can easily be expanded by changing
    the internals slightly or using ``keras.layers.TimeDistributed``.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        num_seeds: int = 1,
        key_dim: int = 32,
        num_heads: int = 4,
        seed_dim: int = None,
        dropout: float = 0.05,
        num_dense_feedforward: int = 2,
        dense_units: int = 128,
        dense_activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        use_bias=True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """Creates a multi-head attention block (MAB) which will perform cross-attention between an input set
        and a set of seed vectors (typically one for a single summary) with summary_dim output dimensions.

        Could also be used as part of a ``DeepSet`` for representing learnable instead of fixed pooling.

        Parameters
        ----------
        ##TODO
        """

        super().__init__(**kwargs)

        self.mab = MultiHeadAttentionBlock(
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_dense_feedforward=num_dense_feedforward,
            output_dim=summary_dim,
            dense_units=dense_units,
            dense_activation=dense_activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )
        self.seed_vector = self.add_weight(
            shape=(num_seeds, seed_dim if seed_dim is not None else key_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
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

    def call(self, set_x: Tensor, **kwargs) -> Tensor:
        """Performs the forward pass through the PMA block.

        Parameters
        ----------
        set_x   : Tensor
            Input of shape (batch_size, set_size, input_dim)

        Returns
        -------
        summary : Tensor
            Output of shape (batch_size, num_seeds * summary_dim)
        """

        set_x_transformed = self.feedforward(set_x, training=kwargs.get("training", False))
        batch_size = ops.shape(set_x)[0]
        seed_vector_expanded = ops.expand_dims(self.seed_vector, axis=0)
        seed_tiled = ops.tile(seed_vector_expanded, [batch_size, 1, 1])
        summaries = self.mab(seed_tiled, set_x_transformed, **kwargs)
        return ops.reshape(summaries, (ops.shape(summaries)[0], -1))
