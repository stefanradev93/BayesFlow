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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from bayesflow.wrappers import SpectralNormalization
from bayesflow.exceptions import ConfigurationError


class TailNetwork(tf.keras.Model):
    """Implements a standard fully connected network for learning the degrees of a latent Student-t distribution."""

    def __init__(self, meta):
        """Creates a network which will adaptively learn the heavy-tailedness of the target distribution.

        Parameters
        ----------
        meta : list(dict)
            A list of dictionaries, where each dictionary holds parameter-value pairs
            for a single :class:`keras.Dense` layer
        """
        
        super(TailNetwork, self).__init__()

        # Create network body
        self.dense = Sequential(
            # Hidden layer structure
            [Dense(**meta['dense_args'])
             for _ in range(meta['n_dense'])]
        )

        # Create network head
        self.dense.add(Dense(1, activation='softplus', **{k: v for 
        k, v in meta['dense_args'].items() if k != 'units' and k != 'activation'}))
        
    def call(self, condition):
        """Performs a forward pass through the tail network. Output is the learned "degrees of freedom" parameter
        for the latent t-distribution.

        Parameters
        ----------
        condition   : tf.Tensor
            the conditioning vector of interest, for instance ``x = summary(x)``, shape (batch_size, summary_dim)
        """

        # Output is bounded between (1, inf)
        out = self.dense(condition) + 1.0
        return out


class Permutation(tf.keras.Model):
    """Implements a layer to permute the inputs entering a (conditional) coupling layer. Uses
    fixed permutations, as these perform equally well compared to learned permutations."""

    def __init__(self, input_dim):
        """Creates an invertible permutation layer for a conditional invertible layer.

        Parameters
        ----------
        input_dim  : int
            Ihe dimensionality of the input to the (conditional) coupling layer.
        """

        super(Permutation, self).__init__()

        permutation_vec = np.random.permutation(input_dim)
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(initial_value=permutation_vec,
                                       trainable=False,
                                       dtype=tf.int32,
                                       name='permutation')
        self.inv_permutation = tf.Variable(initial_value=inv_permutation_vec,
                                           trainable=False,
                                           dtype=tf.int32,
                                           name='inv_permutation')

    def call(self, target, inverse=False):
        """Permutes a batch of target vectors over the last axis.

        Parameters
        ----------
        target   : tf.Tensor of shape (batch_size, ...)
            The target vector to be permuted over its last axis.
        inverse  : bool, default: False
            Controls if the current pass is forward (``inverse=False``) or inverse (``inverse=True``).

        Returns
        -------
        out      : tf.Tensor of the same shape as `target`.
            The permuted target vector.

        """

        if not inverse:
            return tf.transpose(tf.gather(tf.transpose(target), self.permutation))
        return tf.transpose(tf.gather(tf.transpose(target), self.inv_permutation))


class ActNorm(tf.keras.Model):
    """Implements an Activation Normalization (ActNorm) Layer."""

    def __init__ (self, meta, **kwargs):
        """Creates an instance of an ActNorm Layer as proposed by [1].

        Activation Normalization is learned invertible normalization, using
        a Scale (s) and Bias (b) vector [1].
            y = s * x + b (forward)
            x = (y - b)/s (inverse)
        
        The scale and bias can be data dependent initalized, such that the
        output has a mean of zero and standard deviation of one [1,2]. 
        Alternatively, it is initialized with vectors of ones (scale) and 
        zeros (bias).

        [1] - Kingma, Diederik P., and Prafulla Dhariwal. 
              "Glow: Generative flow with invertible 1x1 convolutions." 
               arXiv preprint arXiv:1807.03039 (2018).

        [2] - Salimans, Tim, and Durk P. Kingma. 
              "Weight normalization: A simple reparameterization to accelerate 
               training of deep neural networks." 
              Advances in neural information processing systems 29 
              (2016): 901-909.

        Parameters
        ----------
        meta : dict
            Contains initialization settings for the `ActNorm` layer.
        """

        super(ActNorm, self).__init__(**kwargs)

        # Initialize scale and bias with zeros and ones if no batch for initalization was provided.
        if meta.get('act_norm_init') is None:
            self.scale = tf.Variable(tf.ones((meta['n_params'], )),
                                     trainable=True,
                                     name='act_norm_scale')

            self.bias  = tf.Variable(tf.zeros((meta['n_params'], )),
                                     trainable=True,
                                     name='act_norm_bias')
        else:
            self._initalize_parameters_data_dependent(meta['act_norm_init'])

    def call(self, target, inverse=False):
        """Performs one pass through the actnorm layer (either inverse or forward) and normalizes
        the last axis of `target`.
        
        Parameters
        ----------
        target     : tf.Tensor of shape (batch_size, ...)
            the target variables of interest, i.e., parameters for posterior estimation
        inverse    : bool, optional, default: False
            Flag indicating whether to run the block forward or backwards
        
        Returns
        -------
        (z, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (,)

        target          :  tf.Tensor
            If inverse=True: The inversly transformed targets, shape == target.shape

        Important
        ---------
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``.
        """
        
        if not inverse:
            return self._forward(target)
        else:
            return self._inverse(target)

    def _forward(self, target):
        """Performs a forward pass through the `ActNorm` layer."""

        z = self.scale * target + self.bias
        ldj = tf.math.reduce_sum(tf.math.log(tf.math.abs(self.scale)), axis=-1)
        return z, ldj     

    def _inverse(self, target):
        """Performs an inverse pass through the `ActNorm` layer."""

        return (target - self.bias) / self.scale

    def _initalize_parameters_data_dependent(self, init_data):
        """Performs a data dependent initalization of the scale and bias.
        
        Initalizes the scale and bias vector as proposed by [1], such that the 
        layer output has a mean of zero and a standard deviation of one.

        [1] - Salimans, Tim, and Durk P. Kingma. 
        "Weight normalization: A simple reparameterization to accelerate 
        training of deep neural networks." 
        Advances in neural information processing systems 29 
        (2016): 901-909.

        Parameters
        ----------
        init_data    : tf.Tensor of shape (batch size, number of parameters) 
            Initiall values to estimate the scale and bias parameters by computing 
            the mean and standard deviation along the first dimension of `init_data`.
        """
        
        # 2D Tensor case, assume first batch dimension
        if len(init_data.shape) == 2:
            mean = tf.math.reduce_mean(init_data, axis=0) 
            std  = tf.math.reduce_std(init_data,  axis=0)
        # 3D Tensor case, assume first batch dimension, second number of observations dimension
        elif len(init_data.shape) == 3:
            mean = tf.math.reduce_mean(init_data, axis=(0, 1)) 
            std  = tf.math.reduce_std(init_data,  axis=(0, 1))
        # Raise other cases
        else:
            raise ConfigurationError("""Currently, ActNorm supports only 2D and 3D Tensors, 
                                     but act_norm_init contains data with shape.""".format(init_data.shape))

        scale = 1.0 / std
        bias  = (-1.0 * mean) / std
        
        self.scale = tf.Variable(scale, trainable=True, name='act_norm_scale')
        self.bias  = tf.Variable(bias, trainable=True, name='act_norm_bias')


class DenseCouplingNet(tf.keras.Model):
    """Implements a conditional version of a standard fully connected (FC) network.
    Would also work as an unconditional estimator."""

    def __init__(self, meta, n_out, **kwargs):
        """Creates a conditional coupling net (FC neural network).

        Parameters
        ----------
        meta     : dict
            A dictionary which holds arguments for a dense layer.
        n_out    : int
            Number of outputs of the coupling net
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the `tf.keras.Model` constructor. 
        """

        super(DenseCouplingNet, self).__init__(**kwargs)

        # Create network body (input and hidden layers)
        self.dense = Sequential(
            # Hidden layer structure
            [SpectralNormalization(Dense(**meta['dense_args'])) if meta['spec_norm'] else Dense(**meta['dense_args'])
             for _ in range(meta['n_dense'])]
        )
        # Create network output head
        self.dense.add(Dense(n_out, **{k: v for k, v in meta['dense_args'].items() if k != 'units'}))
        self.dense.build(input_shape=())

    def call(self, target, condition, **kwargs):
        """Concatenates target and condition and performs a forward pass through the coupling net.

        Parameters
        ----------
        target      : tf.Tensor
          The split estimation quntities, for instance, parameters :math:`\\theta \sim p(\\theta)` of interest, shape (batch_size, ...)
        condition   : tf.Tensor or None
            the conditioning vector of interest, for instance ``x = summary(x)``, shape (batch_size, summary_dim)
        """

        # Handle case no condition
        if condition is None:
            return self.dense(target, **kwargs)

        # Handle 3D case for a set-flow
        if len(target.shape) == 3 and len(condition.shape) == 2:
            # Extract information about second dimension
            N = int(target.shape[1])
            condition = tf.stack([condition] * N, axis=1)
        inp = tf.concat((target, condition), axis=-1)
        out = self.dense(inp, **kwargs)
        return out
