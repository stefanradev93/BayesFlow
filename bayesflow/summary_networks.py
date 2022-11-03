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
from tensorflow.keras.layers import Dense, Conv1D, LSTM
from tensorflow.keras.models import Sequential
from bayesflow.helper_functions import build_meta_dict
from bayesflow import default_settings


class InvariantModule(tf.keras.Model):
    """ Implements an invariant module performing a permutation-invariant transform. 
    
    For details and rationale, see:
    
    https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """
    
    def __init__(self, meta):
        super(InvariantModule, self).__init__()
        
        self.s1 = Sequential([Dense(**meta['dense_s1_args']) for _ in range(meta['n_dense_s1'])])
        self.s2 = Sequential([Dense(**meta['dense_s2_args']) for _ in range(meta['n_dense_s2'])])
                    
    def call(self, x):
        """ Performs the forward pass of a learnable invariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim)
        """
        
        x_reduced = tf.reduce_mean(self.s1(x), axis=1)
        out = self.s2(x_reduced)
        return out


class EquivariantModule(tf.keras.Model):
    """Implements an equivariant module performing an equivariant transform. 
    
    For details and justification, see:

    https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """
    
    def __init__(self, meta):
        super(EquivariantModule, self).__init__()
        
        self.invariant_module = InvariantModule(meta)
        self.s3 = Sequential([Dense(**meta['dense_s3_args']) for _ in range(meta['n_dense_s3'])])
                    
    def call(self, x):
        """Performs the forward pass of a learnable equivariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, N, equiv_dim)
        """
        
        # Store N
        N = int(x.shape[1])
        
        # Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module(x)
        out_inv_rep = tf.stack([out_inv] * N, axis=1)
        
        # Concatenate each x with the repeated invariant embedding
        out_c = tf.concat([x, out_inv_rep], axis=-1)
        
        # Pass through equivariant func
        out = self.s3(out_c)
        return out


class InvariantNetwork(tf.keras.Model):
    """Implements an invariant network with keras."""

    def __init__(self, meta={}, **kwargs):
        super(InvariantNetwork, self).__init__(**kwargs)

        meta = build_meta_dict(user_dict=meta,
                               default_setting=default_settings.DEFAULT_SETTING_INVARIANT_NET)
        
        self.equiv_seq = Sequential([EquivariantModule(meta) for _ in range(meta['n_equiv'])])
        self.inv = InvariantModule(meta)
        self.out_layer = Dense(meta['summary_dim'], activation='linear')
        self.summary_dim = meta['summary_dim']
    
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
    """ Implements an inception-inspired 1D convolutional layer using different kernel sizes."""

    def __init__(self, meta, **kwargs):
        """ Creates an inception-like Conv1D layer

        Parameters
        ----------
        meta  : dict
            A dictionary which holds the arguments for the internal `Conv1D` layers.
        """

        super(MultiConv1D, self).__init__(**kwargs)
        
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

        super(MultiConvNetwork, self).__init__(**kwargs)
        
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
        """Performs a forward pass through the network by first passing
        x through the sequence of multi-convolutional layers and then applying 
        the LSTM network.

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

        super(SplitNetwork, self).__init__(**kwargs)

        self.num_splits = num_splits
        self.split_data_configurator = split_data_configurator
        self.networks = [network_type(meta) for _ in range(num_splits)]

    def call(self, x):
        """ Performs the forward pass through the networks.

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
