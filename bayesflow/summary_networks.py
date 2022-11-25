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
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from bayesflow.helper_networks import InvariantModule, EquivariantModule, MultiConv1D
from bayesflow import default_settings as defaults


class InvariantNetwork(tf.keras.Model):
    """Implements a deep permutation-invariant network according to [1] and [2].

    [1] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J. (2017). 
    Deep sets. Advances in neural information processing systems, 30.

    [2] Bloem-Reddy, B., & Teh, Y. W. (2020). 
    Probabilistic Symmetries and Invariant Neural Networks. 
    J. Mach. Learn. Res., 21, 90-1.
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
            dense_s1_args=defaults.DEFAULT_SETTING_DENSE_INVARIANT\
                if dense_s1_args is None else dense_s1_args,
            dense_s2_args=defaults.DEFAULT_SETTING_DENSE_INVARIANT\
                if dense_s2_args is None else dense_s2_args,
            dense_s3_args=defaults.DEFAULT_SETTING_DENSE_INVARIANT\
                if dense_s3_args is None else dense_s3_args,
            pooling_fun=pooling_fun
        )

        # Create equivariant layers and final invariant layer
        self.equiv_layers = Sequential([EquivariantModule(settings) for _ in range(num_equiv)])
        self.inv = InvariantModule(settings)

        # Output layer to output "summary_dim" learned summary statistics
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
        out_equiv = self.equiv_layers(x)

        # Pass through final invariant layer 
        out = self.out_layer(self.inv(out_equiv))

        return out


class SequentialNetwork(tf.keras.Model):
    """Implements a sequence of `MultiConv1D` layers followed by an LSTM network. 
    
    For details and rationale, see [1]:

    [1] Radev, S. T., Graw, F., Chen, S., Mutters, N. T., Eichel, V. M., Bärnighausen, T., & Köthe, U. (2021). 
    OutbreakFlow: Model-based Bayesian inference of disease outbreak dynamics with invertible neural networks 
    and its application to the COVID-19 pandemics in Germany. 
    PLoS computational biology, 17(10), e1009472.

    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009472
    """

    def __init__(self, summary_dim=10, num_conv_layers=2, lstm_units=128, conv_settings=None, **kwargs):
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
            it should contain the followin keys:
            - layer_args      (dict) : arguments for `tf.keras.layers.Conv1D` without kernel_size
            - min_kernel_size (int)  : the minimum kernel size (>= 1)
            - max_kernel_size (int)  : the maximum kernel size
        **kwargs        : dict
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        # Take care of None conv_settings
        if conv_settings is None:
            conv_settings = defaults.DEFAULT_SETTING_MULTI_CONV

        self.net = Sequential([
            MultiConv1D(conv_settings)
            for _ in range(num_conv_layers)
        ])
        
        self.lstm = LSTM(lstm_units)
        self.out_layer = Dense(summary_dim, activation='linear')
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


class SplitNetwork(tf.keras.Model):
    """Implements a vertical stack of networks and concatenates their individual outputs. Allows for splitting
    of data to provide an individual network for each split of the data.
    """

    def __init__(self, num_splits, split_data_configurator, network_type=InvariantNetwork, network_kwargs={}, **kwargs):
        """Creates a composite network of `num_splits` sub-networks of type `network_type`, each with configuration
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
        meta                    : dict, optional, default: {}
            A dictionary containing the configuration for the networks.
        **kwargs
            Optional keyword arguments to be passed to the `tf.keras.Model` superclass.
        """

        super().__init__(**kwargs)

        self.num_splits = num_splits
        self.split_data_configurator = split_data_configurator
        self.networks = [network_type(**network_kwargs) for _ in range(num_splits)]

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
