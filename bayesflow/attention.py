# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Sequential


class MultiHeadAttentionBlock(tf.keras.Model):
    """Implements the MAB block from [1] which represents learnable cross-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(self, input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm, **kwargs):
        """Creates a multihead attention block which will typically be used as part of a
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
        use_layer_norm      : boolean
            Whether layer normalization before and after attention + feedforward
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        self.att = MultiHeadAttention(**attention_settings)
        self.ln_pre = LayerNormalization() if use_layer_norm else None
        self.fc = Sequential([Dense(**dense_settings) for _ in range(num_dense_fc)])
        self.fc.add(Dense(input_dim))
        self.ln_post = LayerNormalization() if use_layer_norm else None

    def call(self, x, y, **kwargs):
        """Performs the forward pass through the attention layer.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, set_size_x, input_dim)
        y : tf.Tensor
            Input of shape (batch_size, set_size_y, input_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, set_size_x, input_dim)
        """

        h = x + self.att(x, y, y, **kwargs)
        if self.ln_pre is not None:
            h = self.ln_pre(h, **kwargs)
        out = h + self.fc(h, **kwargs)
        if self.ln_post is not None:
            out = self.ln_post(out, **kwargs)
        return out


class SelfAttentionBlock(tf.keras.Model):
    """Implements the SAB block from [1] which represents learnable self-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(self, input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm, **kwargs):
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
        use_layer_norm      : boolean
            Whether layer normalization before and after attention + feedforward
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        self.mab = MultiHeadAttentionBlock(input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm)

    def call(self, x, **kwargs):
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, set_size, input_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        return self.mab(x, x, **kwargs)


class InducedSelfAttentionBlock(tf.keras.Model):
    """Implements the ISAB block from [1] which represents learnable self-attention specifically
    designed to deal with large sets via a learnable set of "inducing points".

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self, input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm, num_inducing_points, **kwargs
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
        use_layer_norm      : boolean
            Whether layer normalization before and after attention + feedforward
        num_inducing_points : int
            The number of inducing points. Should be lower than the smallest set size
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        init = tf.keras.initializers.GlorotUniform()
        self.I = tf.Variable(init(shape=(num_inducing_points, input_dim)), name="I", trainable=True)
        self.mab0 = MultiHeadAttentionBlock(input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm)
        self.mab1 = MultiHeadAttentionBlock(input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm)

    def call(self, x, **kwargs):
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, set_size, input_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        batch_size = x.shape[0]
        h = self.mab0(tf.stack([self.I] * batch_size), x, **kwargs)
        return self.mab1(x, h, **kwargs)


class PoolingWithAttention(tf.keras.Model):
    """Implements the pooling with multihead attention (PMA) block from [1] which represents
    a permutation-invariant encoder for set-based inputs.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self, summary_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm, num_seeds=1, **kwargs
    ):
        """Creates a multihead attention block (MAB) which will perform cross-attention between an input set
        and a set of seed vectors (typically one for a single summary) with summary_dim output dimensions.

        Could also be used as part of a ``DeepSet`` for representing learnabl instead of fixed pooling.

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
        use_layer_norm      : boolean
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
            summary_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm, **kwargs
        )
        init = tf.keras.initializers.GlorotUniform()
        self.seed_vec = init(shape=(num_seeds, summary_dim))
        self.fc = Sequential([Dense(**dense_settings) for _ in range(num_dense_fc)])
        self.fc.add(Dense(summary_dim))

    def call(self, x, **kwargs):
        """Performs the forward pass through the PMA block.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, set_size, input_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, num_seeds * summary_dim)
        """

        batch_size = x.shape[0]
        out = self.fc(x)
        out = self.mab(tf.stack([self.seed_vec] * batch_size), out, **kwargs)
        return tf.reshape(out, (out.shape[0], -1))
