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

from functools import partial

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, LSTM
from tensorflow.keras.models import Sequential

from bayesflow.helper_functions import build_meta_dict
from bayesflow.helper_networks import InvariantModule, EquivariantModule
from bayesflow import default_settings as defaults


class InvariantNetwork(tf.keras.Model):
    """Implements a deep permutation-invariant network according to [1].
    
    [1] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J. (2017). 
    Deep sets. Advances in neural information processing systems, 30.
    """

    def __init__(self, summary_dim=10, num_dense_s1=2, num_dense_s2=2, num_dense_s3=2, num_equiv=2, 
                 dense_s1_args=None, dense_s2_args=None, dense_s3_args=None, pooling_fun='mean', **kwargs):
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
        dense_s1_args : dict or None, optional, default: None
            The arguments for the dense layers of s1 (inner, pre-pooling function). If `None`,
            defaults will be used (see `default_settings`).
        dense_s2_args : dict or None, optional, default: None
            The arguments for the dense layers of s2 (outer, post-pooling function). If `None`,
            defaults will be used (see `default_settings`).
        dense_s3_args : dict or None, optional, default: None
            The arguments for the dense layers of s3 (equivariant function). If `None`,
            defaults will be used (see `default_settings`).
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
            dense_s1_args=defaults.DEFAULT_SETTING_DENSE_INVARIANT\
                if dense_s1_args is None else dense_s1_args,
            dense_s2_args=defaults.DEFAULT_SETTING_DENSE_INVARIANT\
                if dense_s2_args is None else dense_s1_args,
            dense_s3_args=defaults.DEFAULT_SETTING_DENSE_INVARIANT\
                if dense_s3_args is None else dense_s1_args,
            pooling_fun=pooling_fun
        )

        self.equiv_seq = Sequential([EquivariantModule(settings) for _ in range(num_equiv)])
        self.inv = InvariantModule(settings)
        self.out_layer = Dense(summary_dim, activation='linear')
        self.summary_dim = summary_dim

    def call(self, x):
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
        out_equiv = self.equiv_seq(x)

        # Pass through final invariant layer 
        out = self.out_layer(self.inv(out_equiv))

        return out


class MultiConv1D(tf.keras.Model):
    """Implements an inception-inspired 1D convolutional layer using different kernel sizes."""

    def __init__(self, meta, **kwargs):
        """ Creates an inception-like Conv1D layer

        Parameters
        ----------
        meta  : dict
            A dictionary which holds the arguments for the internal `Conv1D` layers.
        """

        super().__init__(**kwargs)
        
        # Create a list of Conv1D layers with different kernel sizes
        # ranging from 'min_kernel_size' to 'max_kernel_size'
        self.convs = [
            Conv1D(kernel_size=f, **meta['layer_args'])
            for f in range(meta['min_kernel_size'], meta['max_kernel_size'])
        ]

        # Create final Conv1D layer for dimensionalitiy reduction
        dim_red_args = {k : v for k, v in meta['layer_args'].items() if k not in ['kernel_size', 'strides']}
        dim_red_args['kernel_size'] = 1
        dim_red_args['strides'] = 1
        self.dim_red = Conv1D(**dim_red_args)
        
    def call(self, x, **kwargs):
        """ Performs a forward pass through the layer.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_time_steps, n_time_series)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, n_time_steps, n_filters)
        """
        
        out = tf.concat([conv(x, **kwargs) for conv in self.convs], axis=-1)
        out = self.dim_red(out, **kwargs)
        return out


class MultiConvNetwork(tf.keras.Model):
    """Implements a sequence of `MultiConv1D` layers followed by an LSTM network. 
    
    For details and rationale, see:

    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009472
    """

    def __init__(self, meta={}, **kwargs):
        """ Creates a stack of inception-like layers followed by an LSTM network, with the idea
        of learning vector representations from multivariate time series data.

        Parameters
        ----------
        meta  : dict
            A dictionary which holds the arguments for the `MultiConv1D` and `LSTM` layers.
        """

        super().__init__(**kwargs)
        
        meta = build_meta_dict(user_dict=meta,
                        default_setting=default_settings.DEFAULT_SETTING_MULTI_CONV_NET)

        self.net = Sequential([
            MultiConv1D(meta['conv_args'])
            for _ in range(meta['n_conv_layers'])
        ])
        
        self.lstm = LSTM(**meta['lstm_args'])
        self.out_layer = Dense(meta['summary_dim'], activation='linear')
        self.summary_dim = meta['summary_dim']

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


class SplitNetwork(tf.keras.Model):
    """Implements a vertical stack of networks and concatenates their individual outputs. Allows for splitting
    of data to provide an individual network for each split of the data.
    """

    def __init__(self, num_splits, split_data_configurator, network_type=InvariantNetwork, meta={}, **kwargs):
        """ Creates a composite network of `num_splits` sub-networks of type `network_type`, each with configuration
        specified by `meta`.

        Parameters
        ----------
        num_splits              : int
            The number if splits for the data, which will equal the number of sub-networks.
        split_data_configurator : callable
            Function that takes the arguments `i` and `x` where `i` is the index of the network
            and `x` are the inputs to the `SplitNetwork`. Should return the input for the corresponding network.
            
            For example, to achieve a network with is permutation-invariant both vertically (i.e., across rows)
            and horizontally (i.e., across columns), one could to:
            `def config(i, x):
            TODO
            `
        network_type            : callable, optional, default: `InvariantNetowk`
            Type of neural network to use.
        meta                    : dict, optional, default: {}
            A dictionary containing the configuration for the networks.
        **kwargs
            Optional keyword arguments to be passed to the `tf.keras.Model` superclass.
        """

        super().__init__(**kwargs)

        self.num_splits = num_splits
        self.split_data_configurator = split_data_configurator
        self.networks = [network_type(meta) for _ in range(num_splits)]

    def call(self, x):
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

        out = [self.networks[i](self.split_data_configurator(i, x)) for i in range(self.num_splits)]
        out = tf.concat(out, axis = -1)
        return out
