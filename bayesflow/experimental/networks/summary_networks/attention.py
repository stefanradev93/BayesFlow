
import keras
from keras import ops

from bayesflow.experimental.types import Tensor


class MultiHeadAttentionBlock(keras.layers.Layer):
    """Implements the MAB block from [1] which represents learnable cross-attention.

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

        self.att = keras.layers.MultiHeadAttention(**attention_settings)
        self.ln_pre = keras.layers.LayerNormalization() if layer_norm else None
        self.fc = keras.Sequential([keras.layers.Dense(**dense_settings) for _ in range(num_dense_fc)])
        self.fc.add(keras.layers.Dense(input_dim))
        self.ln_post = keras.layers.LayerNormalization() if layer_norm else None

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


class SelfAttentionBlock(keras.layers.Layer):
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


class InducedSelfAttentionBlock(keras.layers.Layer):
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


class PoolingWithAttention(keras.layers.Layer):
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
        summary_dim: int,
        attention_settings: dict,
        num_dense_fc: int,
        dense_settings: dict,
        layer_norm: bool,
        num_seeds: int = 1,
        **kwargs
    ):
        """Creates a multi-head attention block (MAB) which will perform cross-attention between an input set
        and a set of seed vectors (typically one for a single summary) with summary_dim output dimensions.

        Could also be used as part of a ``DeepSet`` for representing learnable instead of fixed pooling.

        Parameters
        ----------
        summary_dim         : int
            The dimensionality of the learned permutation-invariant representation.
        attention_settings  : dict
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention.
        num_dense_fc        : int
            The number of hidden layers for the internal feedforward network
        dense_settings      : dict
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer
        layer_norm          : boolean
            Whether layer normalization before and after attention + feedforward
        num_seeds           : int, optional, default: 1
            The number of "seed vectors" to use. Each seed vector represents a permutation-invariant
            summary of the entire set. If you use ``num_seeds > 1``, the resulting seeds will be flattened
            into a 2-dimensional output, which will have a dimensionality of ``num_seeds * summary_dim``
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        self.mab = MultiHeadAttentionBlock(
            summary_dim, attention_settings, num_dense_fc, dense_settings, layer_norm, **kwargs
        )
        self.seed_vector = self.add_weight(
            shape=(num_seeds, summary_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.fc = keras.Sequential([keras.layers.Dense(**dense_settings) for _ in range(num_dense_fc)])
        self.fc.add(keras.layers.Dense(summary_dim))

    def call(self, x: Tensor, **kwargs):
        """Performs the forward pass through the PMA block.

        Parameters
        ----------
        x   : Tensor
            Input of shape (batch_size, set_size, input_dim)

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, num_seeds * summary_dim)
        """

        out = self.fc(x)
        batch_size = ops.shape(x)[0]
        seed_vector_expanded = ops.expand_dims(self.seed_vector, axis=0)
        seed_tiled = ops.tile(seed_vector_expanded, [batch_size, 1, 1])
        out = self.mab(seed_tiled, out, **kwargs)
        return ops.reshape(out, (ops.shape(out)[0], -1))
