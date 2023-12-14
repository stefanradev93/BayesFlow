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

from warnings import warn

import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential

from bayesflow import default_settings as defaults
from bayesflow.attention import (
    InducedSelfAttentionBlock,
    MultiHeadAttentionBlock,
    PoolingWithAttention,
    SelfAttentionBlock,
)
from bayesflow.helper_networks import EquivariantModule, InvariantModule, MultiConv1D


class TimeSeriesTransformer(tf.keras.Model):
    """Implements a many-to-one transformer architecture for time series encoding.
    Some ideas can be found in [1]:

    [1] Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2022).
    Transformers in time series: A survey. arXiv preprint arXiv:2202.07125.
    https://arxiv.org/abs/2202.07125
    """

    def __init__(
        self,
        input_dim,
        attention_settings=None,
        dense_settings=None,
        use_layer_norm=True,
        num_dense_fc=2,
        summary_dim=10,
        num_attention_blocks=2,
        template_type="lstm",
        bidirectional=False,
        template_dim=64,
        **kwargs,
    ):
        """Creates a transformer architecture for encoding time series data into fixed size vectors given by
        ``summary_dim``. It features a recurrent network given by ``template_type`` which is responsible for
        providing a single summary of the time series which then attends to each point in the time series pro-
        cessed via a series of ``num_attention_blocks`` self-attention layers.

        Important: Assumes that positional encodings have been appended to the input time series, e.g.,
        through a custom configurator.

        Recommended: When using transformers as summary networks, you may want to use a smaller learning rate
        during training, e.g., setting ``default_lr=5e-5`` in a ``Trainer`` instance.

        Layer normalization (controllable through the ``use_layer_norm`` keyword argument) may not always work
        well in certain applications. Consider setting it to ``False`` if the network is underperforming.

        Parameters
        ----------
        input_dim            : int
            The dimensionality of the input data (last axis).
        attention_settings   : dict or None, optional, default None
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer.
            If ``None``, default settings will be used (see ``bayesflow.default_settings``)
            For instance, to use an attention block with 4 heads and key dimension 32, you can do:

            ``attention_settings=dict(num_heads=4, key_dim=32)``

            You may also want to include dropout regularization in small-to-medium data regimes:

            ``attention_settings=dict(num_heads=4, key_dim=32, dropout=0.1)``

            For more details and arguments, see:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        dense_settings       : dict or None, optional, default: None
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer.
            For instance, to use hidden layers with 32 units and a relu activation, you can do:

            ``dict(units=32, activation='relu')

            For more details and arguments, see:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        use_layer_norm       : boolean, optional, default: True
            Whether layer normalization before and after attention + feedforward
        num_dense_fc         : int, optional, default: 2
            The number of hidden layers for the internal feedforward network
        summary_dim          : int
            The dimensionality of the learned permutation-invariant representation.
        num_attention_blocks : int, optional, default: 2
            The number of self-attention blocks to use before pooling.
        template_type        : str or callable, optional, default: 'lstm'
            The many-to-one (learnable) transformation of the time series.
            if ``lstm``, an LSTM network will be used.
            if ``gru``, a GRU unit will be used.
            if callable, a reference to ``template_type`` will be stored as an attribute.
        bidirectional        : bool, optional, default: False
            Indicates whether the involved LSTM template network is bidirectional (i.e., forward
            and backward in time) or unidirectional (forward in time). Defaults to False, but may
            increase performance in some applications.
        template_dim         : int, optional, default: 64
            Only used if ``template_type`` in ['lstm', 'gru']. The number of hidden
            units (equiv. output dimensions) of the recurrent network.
            When using ``bidirectional=True``, the output dimensions of the template
            will be double the template_dim size, so consider reducing it in half.
        **kwargs             : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        # Process internal attention settings
        if attention_settings is None:
            attention_settings = defaults.DEFAULT_SETTING_ATTENTION
        if dense_settings is None:
            dense_settings = defaults.DEFAULT_SETTING_DENSE_ATTENTION

        # Construct a series of self-attention blocks, these will process
        # the time series in a many-to-many fashion
        self.attention_blocks = Sequential()
        for _ in range(num_attention_blocks):
            block = SelfAttentionBlock(input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm)
            self.attention_blocks.add(block)

        # Construct final attention layer, which will perform cross-attention
        # between the outputs ot the self-attention layers and the dynamic template
        if bidirectional:
            final_input_dim = template_dim * 2
        else:
            final_input_dim = template_dim
        self.output_attention = MultiHeadAttentionBlock(
            final_input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm
        )

        # A recurrent network will learn the dynamic many-to-one template
        if template_type.upper() == "LSTM":
            self.template_net = Bidirectional(LSTM(template_dim)) if bidirectional else LSTM(template_dim)
        elif template_type.upper() == "GRU":
            self.template_net = Bidirectional(GRU(template_dim)) if bidirectional else GRU(template_dim)
        else:
            assert callable(template_type), "Argument `template_dim` should be callable or in ['lstm', 'gru']"
            self.template_net = template_type

        # Final output reduces representation into a vector of length summary_dim
        self.output_layer = Dense(summary_dim)
        self.summary_dim = summary_dim

    def call(self, x, **kwargs):
        """Performs the forward pass through the transformer.

        Parameters
        ----------
        x   : tf.Tensor
            Time series input of shape (batch_size, num_time_points, input_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, summary_dim)
        """

        rep = self.attention_blocks(x, **kwargs)
        template = self.template_net(x, **kwargs)
        rep = self.output_attention(tf.expand_dims(template, axis=1), rep, **kwargs)
        rep = tf.squeeze(rep, axis=1)
        out = self.output_layer(rep)
        return out


class SetTransformer(tf.keras.Model):
    """Implements the set transformer architecture from [1] which ultimately represents
    a learnable permutation-invariant function. Designed to naturally model interactions in
    the input set, which may be hard to capture with the simpler ``DeepSet`` architecture.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        input_dim,
        attention_settings=None,
        dense_settings=None,
        use_layer_norm=False,
        num_dense_fc=2,
        summary_dim=10,
        num_attention_blocks=2,
        num_inducing_points=32,
        num_seeds=1,
        **kwargs,
    ):
        """Creates a set transformer architecture according to [1] which will extract permutation-invariant
        features from an input set using a set of seed vectors (typically one for a single summary) with ``summary_dim``
        output dimensions.

        Recommended: When using transformers as summary networks, you may want to use a smaller learning rate
        during training, e.g., setting ``default_lr=1e-4`` in a ``Trainer`` instance.

        Parameters
        ----------
        input_dim            : int
            The dimensionality of the input data (last axis).
        attention_settings   : dict or None, optional, default: None
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer
            For instance, to use an attention block with 4 heads and key dimension 32, you can do:

            ``attention_settings=dict(num_heads=4, key_dim=32)``

            You may also want to include stronger dropout regularization in small-to-medium data regimes:

            ``attention_settings=dict(num_heads=4, key_dim=32, dropout=0.1)``

            For more details and arguments, see:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        dense_settings       : dict or None, optional, default: None
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer.
            For instance, to use hidden layers with 32 units and a relu activation, you can do:

            ``dict(units=32, activation='relu')

            For more details and arguments, see:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        use_layer_norm       : boolean, optional, default: False
            Whether to use layer normalization before and after attention + feedforward
        num_dense_fc         : int, optional, default: 2
            The number of hidden layers for the internal feedforward network
        summary_dim          : int
            The dimensionality of the learned permutation-invariant representation.
        num_attention_blocks : int, optional, default: 2
            The number of self-attention blocks to use before pooling.
        num_inducing_points  : int or None, optional, default: 32
            The number of inducing points. Should be lower than the smallest set size.
            If ``None`` selected, a vanilla self-attention block (SAB) will be used, otherwise
            ISAB blocks will be used. For ``num_attention_blocks > 1``, we currently recommend
            always using some number of inducing points.
        num_seeds            : int, optional, default: 1
            The number of "seed vectors" to use. Each seed vector represents a permutation-invariant
            summary of the entire set. If you use ``num_seeds > 1``, the resulting seeds will be flattened
            into a 2-dimensional output, which will have a dimensionality of ``num_seeds * summary_dim``.
        **kwargs             : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        # Process internal attention settings
        if attention_settings is None:
            attention_settings = defaults.DEFAULT_SETTING_ATTENTION
        if dense_settings is None:
            dense_settings = defaults.DEFAULT_SETTING_DENSE_ATTENTION

        # Construct a series of self-attention blocks
        self.attention_blocks = Sequential()
        for _ in range(num_attention_blocks):
            if num_inducing_points is not None:
                block = InducedSelfAttentionBlock(
                    input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm, num_inducing_points
                )
            else:
                block = SelfAttentionBlock(input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm)
            self.attention_blocks.add(block)

        # Pooler will be applied to the representations learned through self-attention
        self.pooler = PoolingWithAttention(
            summary_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm, num_seeds
        )

        self.summary_dim = summary_dim

    def call(self, x, **kwargs):
        """Performs the forward pass through the set-transformer.

        Parameters
        ----------
        x   : tf.Tensor
            The input set of shape (batch_size, set_size, input_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, summary_dim * num_seeds)
        """

        out = self.attention_blocks(x, **kwargs)
        out = self.pooler(out, **kwargs)
        return out


class DeepSet(tf.keras.Model):
    """Implements a deep permutation-invariant network according to [1] and [2].

    [1] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J. (2017).
    Deep sets. Advances in neural information processing systems, 30.

    [2] Bloem-Reddy, B., & Teh, Y. W. (2020).
    Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1.
    """

    def __init__(
        self,
        summary_dim=10,
        num_dense_s1=2,
        num_dense_s2=2,
        num_dense_s3=2,
        num_equiv=2,
        dense_s1_args=None,
        dense_s2_args=None,
        dense_s3_args=None,
        pooling_fun="mean",
        **kwargs,
    ):
        """Creates a stack of 'num_equiv' equivariant layers followed by a final invariant layer.

        Parameters
        ----------
        summary_dim   : int, optional, default: 10
            The number of learned summary statistics.
        num_dense_s1  : int, optional, default: 2
            The number of dense layers in the inner function of a deep set.
        num_dense_s2  : int, optional, default: 2
            The number of dense layers in the outer function of a deep set.
        num_dense_s3  : int, optional, default: 2
            The number of dense layers in an equivariant layer.
        num_equiv     : int, optional, default: 2
            The number of equivariant layers in the network.
        dense_s1_args : dict or None, optional, default: None
            The arguments for the dense layers of s1 (inner, pre-pooling function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        dense_s2_args : dict or None, optional, default: None
            The arguments for the dense layers of s2 (outer, post-pooling function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        dense_s3_args : dict or None, optional, default: None
            The arguments for the dense layers of s3 (equivariant function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        pooling_fun   : str of callable, optional, default: 'mean'
            If string argument provided, should be one in ['mean', 'max']. In addition, ac actual
            neural network can be passed for learnable pooling.
        **kwargs      : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model.
        """

        super().__init__(**kwargs)

        # Prepare settings dictionary
        settings = dict(
            num_dense_s1=num_dense_s1,
            num_dense_s2=num_dense_s2,
            num_dense_s3=num_dense_s3,
            dense_s1_args=defaults.DEFAULT_SETTING_DENSE_DEEP_SET if dense_s1_args is None else dense_s1_args,
            dense_s2_args=defaults.DEFAULT_SETTING_DENSE_DEEP_SET if dense_s2_args is None else dense_s2_args,
            dense_s3_args=defaults.DEFAULT_SETTING_DENSE_DEEP_SET if dense_s3_args is None else dense_s3_args,
            pooling_fun=pooling_fun,
        )

        # Create equivariant layers and final invariant layer
        self.equiv_layers = Sequential([EquivariantModule(settings) for _ in range(num_equiv)])
        self.inv = InvariantModule(settings)

        # Output layer to output "summary_dim" learned summary statistics
        self.out_layer = Dense(summary_dim, activation="linear")
        self.summary_dim = summary_dim

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable deep invariant transformation consisting of
        a sequence of equivariant transforms followed by an invariant transform.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_obs, data_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim)
        """

        # Pass through series of augmented equivariant transforms
        out_equiv = self.equiv_layers(x, **kwargs)

        # Pass through final invariant layer
        out = self.out_layer(self.inv(out_equiv, **kwargs), **kwargs)

        return out


class InvariantNetwork(DeepSet):
    """Deprecated class for ``InvariantNetwork``."""

    def __init_subclass__(cls, **kwargs):
        warn(
            f"{cls.__name__} will be deprecated at some point. Use ``DeepSet`` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        warn(
            f"{self.__class__.__name__} will be deprecated. at some point. Use ``DeepSet`` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class SequenceNetwork(tf.keras.Model):
    """Implements a sequence of `MultiConv1D` layers followed by an (bidirectional) LSTM network.

    For details and rationale, see [1]:

    [1] Radev, S. T., Graw, F., Chen, S., Mutters, N. T., Eichel, V. M., Bärnighausen, T., & Köthe, U. (2021).
    OutbreakFlow: Model-based Bayesian inference of disease outbreak dynamics with invertible neural networks
    and its application to the COVID-19 pandemics in Germany.
    PLoS computational biology, 17(10), e1009472.

    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009472
    """

    def __init__(
        self, summary_dim=10, num_conv_layers=2, lstm_units=128, bidirectional=False, conv_settings=None, **kwargs
    ):
        """Creates a stack of inception-like layers followed by an LSTM network, with the idea
        of learning vector representations from multivariate time series data.

        Parameters
        ----------
        summary_dim     : int, optional, default: 10
            The number of learned summary statistics.
        num_conv_layers : int, optional, default: 2
            The number of convolutional layers to use.
        lstm_units      : int, optional, default: 128
            The number of hidden LSTM units.
        conv_settings   : dict or None, optional, default: None
            The arguments passed to the `MultiConv1D` internal networks. If `None`,
            defaults will be used from `default_settings`. If a dictionary is provided,
            it should contain the following keys:
            - layer_args      (dict) : arguments for `tf.keras.layers.Conv1D` without kernel_size
            - min_kernel_size (int)  : the minimum kernel size (>= 1)
            - max_kernel_size (int)  : the maximum kernel size
        bidirectional   : bool, optional, default: False
            Indicates whether the involved LSTM network is bidirectional (forward and backward in time)
            or unidirectional (forward in time). Defaults to False, but may increase performance.
        **kwargs        : dict
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        # Take care of None conv_settings
        if conv_settings is None:
            conv_settings = defaults.DEFAULT_SETTING_MULTI_CONV

        self.net = Sequential([MultiConv1D(conv_settings) for _ in range(num_conv_layers)])

        self.lstm = Bidirectional(LSTM(lstm_units)) if bidirectional else LSTM(lstm_units)
        self.out_layer = Dense(summary_dim, activation="linear")
        self.summary_dim = summary_dim

    def call(self, x, **kwargs):
        """Performs a forward pass through the network by first passing `x` through the sequence of
        multi-convolutional layers and then applying the LSTM network.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_time_steps, n_time_series)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, summary_dim)
        """

        out = self.net(x, **kwargs)
        out = self.lstm(out, **kwargs)
        out = self.out_layer(out, **kwargs)
        return out


class SequentialNetwork(SequenceNetwork):
    """Deprecated class for amortizer posterior estimation."""

    def __init_subclass__(cls, **kwargs):
        warn(f"{cls.__name__} will be deprecated. Use `SequenceNetwork` instead.", DeprecationWarning, stacklevel=2)
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        warn(
            f"{self.__class__.__name__} will be deprecated. Use `SequenceNetwork` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class SplitNetwork(tf.keras.Model):
    """Implements a vertical stack of networks and concatenates their individual outputs. Allows for splitting
    of data to provide an individual network for each split of the data.
    """

    def __init__(self, num_splits, split_data_configurator, network_type=DeepSet, network_kwargs={}, **kwargs):
        """Creates a composite network of `num_splits` subnetworks of type `network_type`, each with configuration
        specified by `meta`.

        Parameters
        ----------
        num_splits              : int
            The number if splits for the data, which will equal the number of sub-networks.
        split_data_configurator : callable
            Function that takes the arguments `i` and `x` where `i` is the index of the
            network and `x` are the inputs to the `SplitNetwork`. Should return the input
            for the corresponding network.

            For example, to achieve a network with is permutation-invariant both
            vertically (i.e., across rows)  and horizontally (i.e., across columns), one could to:
            `def split(i, x):
                selector = tf.where(x[:,:,0]==i, 1.0, 0.0)
                selected = x[:,:,1] * selector
                split_x = tf.stack((selector, selected), axis=-1)
                return split_x
            `
            where `x[:,:,0]` contains an integer indicating which split the data
            in `x[:,:,1]` belongs to. All values in `x[:,:,1]` that are not selected
            are set to zero. The selector is passed along with the modified data,
            indicating which rows belong to the split `i`.
        network_type            : callable, optional, default: `InvariantNetowk`
            Type of neural network to use.
        network_kwargs          : dict, optional, default: {}
            A dictionary containing the configuration for the networks.
        **kwargs
            Optional keyword arguments to be passed to the `tf.keras.Model` superclass.
        """

        super().__init__(**kwargs)

        self.num_splits = num_splits
        self.split_data_configurator = split_data_configurator
        self.networks = [network_type(**network_kwargs) for _ in range(num_splits)]

    def call(self, x, **kwargs):
        """Performs a forward pass through the subnetworks and concatenates their output.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_obs, data_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim)
        """

        out = [self.networks[i](self.split_data_configurator(i, x), **kwargs) for i in range(self.num_splits)]
        out = tf.concat(out, axis=-1)
        return out


class HierarchicalNetwork(tf.keras.Model):
    """Implements a hierarchical summary network according to [1].

    [1] Elsemüller, L., Schnuerch, M., Bürkner, P. C., & Radev, S. T. (2023).
        A Deep Learning Method for Comparing Bayesian Hierarchical Models.
        arXiv preprint arXiv:2301.11873.
    """

    def __init__(self, networks_list, **kwargs):
        """Creates a hierarchical network consisting of stacked summary networks (one for each hierarchical level)
        that are aligned with the probabilistic structure of the processed data.

        Note: The networks will start processing from the lowest hierarchical level (e.g., observational level)
        up to the highest hierarchical level. It is recommended to provide higher-level networks with more
        expressive power to allow for an adequate compression of lower-level data.

        Example: For two-level hierarchical models with the assumption of temporal dependencies on the lowest
        hierarchical level (e.g., observational level) and exchangeable units at the higher level
        (e.g., group level), a list of [SequenceNetwork(), DeepSet()] could be passed.

        ----------

        Parameters:
        networks_list : list of tf.keras.Model:
            The list of summary networks (one per hierarchical level), starting from the lowest hierarchical level
        """

        super().__init__(**kwargs)
        self.networks = networks_list

    def call(self, x, return_all=False, **kwargs):
        """Performs the forward pass through the hierarchical network,
        transforming the nested input into learned summary statistics.

        Parameters
        ----------
        x          : tf.Tensor of shape (batch_size, ..., data_dim)
            Example, hierarchical data sets with two levels:
            (batch_size, D, L, x_dim) -> reduces to (batch_size, out_dim).
        return_all : boolean, optional, default: False
            Whether to return all intermediate outputs (True) or just
            the final one (False).

        Returns
        -------
        out : tf.Tensor
            Output of shape ``(batch_size, out_dim) if return_all=False`` else a tuple
            of ``len(outputs) == len(networks)`` corresponding to all outputs of all networks.
        """

        if return_all:
            outputs = []
            for net in self.networks:
                x = net(x, **kwargs)
                outputs.append(x)
            return outputs
        else:
            for net in self.networks:
                x = net(x, **kwargs)
            return x
