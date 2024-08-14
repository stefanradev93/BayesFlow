import keras
from keras.saving import register_keras_serializable

from bayesflow.types import Tensor

from ..summary_network import SummaryNetwork

from .sab import SetAttentionBlock
from .isab import InducedSetAttentionBlock
from .pma import PoolingByMultiHeadAttention


@register_keras_serializable(package="bayesflow.networks")
class SetTransformer(SummaryNetwork):
    """Implements the set transformer architecture from [1] which ultimately represents
    a learnable permutation-invariant function. Designed to naturally model interactions in
    the input set, which may be hard to capture with the simpler ``DeepSet`` architecture.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.

    Note: Currently works only on 3D inputs but can easily be expanded by changing
    the internals slightly or using ``keras.layers.TimeDistributed``.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        num_attention_blocks: int = 2,
        num_inducing_points: int = None,
        num_seeds: int = 1,
        key_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.05,
        num_dense_feedforward: int = 2,
        dense_units: int = 128,
        dense_activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        use_bias=True,
        layer_norm: bool = True,
        set_attention_output_dim: int = None,
        seed_dim: int = None,
        **kwargs,
    ):
        """
        #TODO
        """

        super().__init__(**kwargs)

        # Construct a series of set-attention blocks
        self.attention_blocks = keras.Sequential()

        attention_block_settings = dict(
            num_inducing_points=num_inducing_points,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_dense_feedforward=num_dense_feedforward,
            output_dim=set_attention_output_dim,
            dense_units=dense_units,
            dense_activation=dense_activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )

        for _ in range(num_attention_blocks):
            if num_inducing_points is not None:
                block = InducedSetAttentionBlock(**attention_block_settings)
            else:
                block = SetAttentionBlock(**{k: v for k, v in attention_block_settings.items() if k != "num_inducing_points"})
            self.attention_blocks.add(block)

        # Pooling will be applied as a final step to the abstract representations obtained from set attention
        attention_block_settings.pop("num_inducing_points")
        attention_block_settings.pop("output_dim")
        pooling_settings = dict(seed_dim=seed_dim, num_seeds=num_seeds, summary_dim=summary_dim)
        self.pooling_by_attention = PoolingByMultiHeadAttention(**attention_block_settings | pooling_settings)

        # Output projector is needed to keep output dimensions be summary_dim in case of num_seeds > 1
        self.output_projector = keras.layers.Dense(summary_dim)

    def call(self, x: Tensor, **kwargs) -> Tensor:
        """Performs the forward pass through the set-transformer.

        :param x: Tensor of shape (batch_size, set_size, input_dim)

        :param kwargs: Additional keyword arguments to each block

        :return: Tensor of shape (batch_size, output_dim)
        """
        summary = self.attention_blocks(x, **kwargs)
        summary = self.pooling_by_attention(summary, **kwargs)
        summary = self.output_projector(summary)
        return summary
