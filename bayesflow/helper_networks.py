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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout
from tensorflow.keras.models import Sequential

from bayesflow.exceptions import ConfigurationError
from bayesflow.wrappers import SpectralNormalization


class DenseCouplingNet(tf.keras.Model):
    """Implements a conditional version of a standard fully connected (FC) network.
    Would also work as an unconditional estimator."""

    def __init__(self, settings, dim_out, **kwargs):
        """Creates a conditional coupling net (FC neural network).

        Parameters
        ----------
        settings : dict
            A dictionary holding arguments for a dense layer:
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

            As well as custom arguments for settings such as residual networks,
            dropout, and spectral normalization.
        dim_out  : int
            Number of outputs of the coupling net. Determined internally by the
            consumer classes.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the `tf.keras.Model` constructor.
        """

        super().__init__(**kwargs)

        # Create network body (input and hidden layers)
        self.fc = Sequential()
        for _ in range(settings["num_dense"]):
            # Create dense layer with dict kwargs
            layer = Dense(**settings["dense_args"])

            # Wrap in spectral normalization, if specified
            if settings.get("spec_norm") is True:
                layer = SpectralNormalization(layer)
            self.fc.add(layer)

            # Figure out which dropout to use, MC has precedence over standard
            # Fails gently, if no dropout_prob is specified
            # Case both specified, MC wins
            if settings.get("dropout") and settings.get("mc_dropout"):
                self.fc.add(MCDropout(dropout_prob=settings["dropout_prob"]))

            # Case only dropout, use standard
            elif settings.get("dropout") and not settings.get("mc_dropout"):
                self.fc.add(Dropout(rate=settings["dropout_prob"]))

            # Case only MC, use MC
            elif not settings.get("dropout") and settings.get("mc_dropout"):
                self.fc.add(MCDropout(dropout_prob=settings["dropout_prob"]))

            # No dropout
            else:
                pass

        # Set residual flag
        if settings.get("residual"):
            self.fc.add(Dense(dim_out, **{k: v for k, v in settings["dense_args"].items() if k != "units"}))
            self.residual_output = Dense(dim_out, kernel_initializer="zeros")
        else:
            self.fc.add(Dense(dim_out, kernel_initializer="zeros"))
            self.residual_output = None

        self.fc.build(input_shape=())

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
            if self.residual_output is not None:
                return self.residual_output(self.fc(target, **kwargs) + target, **kwargs)
            else:
                return self.fc(target, **kwargs)

        # Handle 3D case for a set-flow and repeat condition over
        # the second `time` or `n_observations` axis of `target``
        if len(tf.shape(target)) == 3 and len(tf.shape(condition)) == 2:
            shape = tf.shape(target)
            condition = tf.expand_dims(condition, 1)
            condition = tf.tile(condition, [1, shape[1], 1])
        inp = tf.concat((target, condition), axis=-1)
        out = self.fc(inp, **kwargs)

        if self.residual_output is not None:
            out = self.residual_output(out + target, **kwargs)
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

        super().__init__()

        permutation_vec = np.random.permutation(input_dim)
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(
            initial_value=permutation_vec, trainable=False, dtype=tf.int32, name="permutation"
        )
        self.inv_permutation = tf.Variable(
            initial_value=inv_permutation_vec, trainable=False, dtype=tf.int32, name="inv_permutation"
        )

    def call(self, target, inverse=False):
        """Permutes a batch of target vectors over the last axis.

        Parameters
        ----------
        target   : tf.Tensor of shape (batch_size, ...)
            The target vector to be permuted over its last axis.
        inverse  : bool, optional, default: False
            Controls if the current pass is forward (``inverse=False``) or inverse (``inverse=True``).

        Returns
        -------
        out      : tf.Tensor of the same shape as `target`.
            The (un-)permuted target vector.
        """

        if not inverse:
            return self._forward(target)
        else:
            return self._inverse(target)

    def _forward(self, target):
        """Performs a fixed permutation over the last axis."""
        return tf.gather(target, self.permutation, axis=-1)

    def _inverse(self, target):
        """Un-does the fixed permutation over the last axis."""
        return tf.gather(target, self.inv_permutation, axis=-1)


class Orthogonal(tf.keras.Model):
    """Implements a learnable orthogonal transformation according to [1]. Can be
    used as an alternative to a fixed ``Permutation`` layer.

    [1] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1
    convolutions. Advances in neural information processing systems, 31.
    """

    def __init__(self, input_dim):
        """Creates an invertible orthogonal transformation (generalized permutation)

        Parameters
        ----------
        input_dim  : int
            Ihe dimensionality of the input to the (conditional) coupling layer.
        """

        super().__init__()

        init = tf.keras.initializers.Orthogonal()
        self.W = tf.Variable(
            initial_value=init(shape=(input_dim, input_dim)), trainable=True, dtype=tf.float32, name="learnable_permute"
        )

    def call(self, target, inverse=False):
        """Transforms a batch of target vectors over the last axis through an approximately
        orthogonal transform.

        Parameters
        ----------
        target   : tf.Tensor of shape (batch_size, ...)
            The target vector to be rotated over its last axis.
        inverse  : bool, optional, default: False
            Controls if the current pass is forward (``inverse=False``) or inverse (``inverse=True``).

        Returns
        -------
        out      : tf.Tensor of the same shape as `target`.
            The (un-)rotated target vector.
        """

        if not inverse:
            return self._forward(target)
        else:
            return self._inverse(target)

    def _forward(self, target):
        """Performs a learnable generalized permutation over the last axis."""

        shape = tf.shape(target)
        rank = len(shape)
        log_det = tf.math.log(tf.math.abs(tf.linalg.det(self.W)))
        if rank == 2:
            z = tf.linalg.matmul(target, self.W)
        else:
            z = tf.tensordot(target, self.W, [[rank - 1], [0]])
            log_det = tf.cast(shape[1], tf.float32) * log_det
        return z, log_det

    def _inverse(self, z):
        """Un-does the learnable permutation over the last axis."""

        W_inv = tf.linalg.inv(self.W)
        rank = len(tf.shape(z))
        if rank == 2:
            return tf.linalg.matmul(z, W_inv)
        return tf.tensordot(z, W_inv, [[rank - 1], [0]])


class MCDropout(tf.keras.Model):
    """Implements Monte Carlo Dropout as a Bayesian approximation according to [1].

    Perhaps not the best approximation, but arguably the cheapest one out there!

    [1] Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation:
    Representing model uncertainty in deep learning.
    In international conference on machine learning (pp. 1050-1059). PMLR.
    """

    def __init__(self, dropout_prob=0.1, **kwargs):
        """Creates a custom instance of an MC Dropout layer. Will be used both
        during training and inference.

        Parameters
        ----------
        dropout_prob  : float, optional, default: 0.1
            The dropout rate to be passed to ``tf.keras.layers.Dropout()``.
        """

        super().__init__(**kwargs)
        self.drop = Dropout(dropout_prob)

    def call(self, inputs):
        """Randomly sets elements of ``inputs`` to zero.

        Parameters
        ----------
        inputs : tf.Tensor
            Input of shape (batch_size, ...)

        Returns
        -------
        out    : tf.Tensor
            Output of shape (batch_size, ...), same as ``inputs``.

        """

        out = self.drop(inputs, training=True)
        return out


class ActNorm(tf.keras.Model):
    """Implements an Activation Normalization (ActNorm) Layer.
    Activation Normalization is learned invertible normalization, using
    a Scale (s) and Bias (b) vector::

       y = s * x + b (forward)
       x = (y - b)/s (inverse)

    Notes
    -----

    The scale and bias can be data dependent initialized, such that the
    output has a mean of zero and standard deviation of one [1]_[2]_.
    Alternatively, it is initialized with vectors of ones (scale) and
    zeros (bias).

    References
    ----------

    .. [1] Kingma, Diederik P., and Prafulla Dhariwal.
       "Glow: Generative flow with invertible 1x1 convolutions."
       arXiv preprint arXiv:1807.03039 (2018).

    .. [2] Salimans, Tim, and Durk P. Kingma.
       "Weight normalization: A simple reparameterization to accelerate
       training of deep neural networks."
       Advances in neural information processing systems 29 (2016): 901-909.
    """

    def __init__(self, latent_dim, act_norm_init, **kwargs):
        """Creates an instance of an ActNorm Layer as proposed by [1].

        Parameters
        ----------
        latent_dim            : int
            The dimensionality of the latent space (equal to the dimensionality of the target variable)
        act_norm_init         : np.ndarray of shape (num_simulations, num_params) or None, optional, default: None
            Optional data-dependent initialization for the internal ``ActNorm`` layers, as done in [1]. Could be helpful
            for deep invertible networks.
        """

        super().__init__(**kwargs)

        # Initialize scale and bias with zeros and ones if no batch for initalization was provided.
        if act_norm_init is None:
            self.scale = tf.Variable(tf.ones((latent_dim,)), trainable=True, name="act_norm_scale")

            self.bias = tf.Variable(tf.zeros((latent_dim,)), trainable=True, name="act_norm_bias")
        else:
            self._initalize_parameters_data_dependent(act_norm_init)

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
            If inverse=True: The inversely transformed targets, shape == target.shape

        Notes
        -----
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``.
        """

        if not inverse:
            return self._forward(target)
        else:
            return self._inverse(target)

    def _forward(self, target):
        """Performs a forward pass through the layer."""

        z = self.scale * target + self.bias
        ldj = tf.math.reduce_sum(tf.math.log(tf.math.abs(self.scale)), axis=-1)
        return z, ldj

    def _inverse(self, target):
        """Performs an inverse pass through the layer."""

        return (target - self.bias) / self.scale

    def _initalize_parameters_data_dependent(self, init_data):
        """Performs a data dependent initalization of the scale and bias.

        Initalizes the scale and bias vector as proposed by [1], such that the
        layer output has a mean of zero and a standard deviation of one.

        [1] - Salimans, Tim, and Durk P. Kingma.
        Weight normalization: A simple reparameterization to accelerate
        training of deep neural networks.
        Advances in neural information processing systems 29
        (2016): 901-909.

        Parameters
        ----------
        init_data    : tf.Tensor of shape (batch size, number of parameters)
            Initiall values to estimate the scale and bias parameters by computing
            the mean and standard deviation along the first dimension of `init_data`.
        """

        # 2D Tensor case, assume first batch dimension
        if tf.rank(init_data) == 2:
            mean = tf.math.reduce_mean(init_data, axis=0)
            std = tf.math.reduce_std(init_data, axis=0)
        # 3D Tensor case, assume first batch dimension, second number of observations dimension
        elif tf.rank(init_data) == 3:
            mean = tf.math.reduce_mean(init_data, axis=(0, 1))
            std = tf.math.reduce_std(init_data, axis=(0, 1))
        # Raise other cases
        else:
            raise ConfigurationError(
                f"""Currently, ActNorm supports only 2D and 3D Tensors,
                                     but act_norm_init contains data with shape {init_data.shape}."""
            )

        scale = 1.0 / std
        bias = (-1.0 * mean) / std

        self.scale = tf.Variable(scale, trainable=True, name="act_norm_scale")
        self.bias = tf.Variable(bias, trainable=True, name="act_norm_bias")


class InvariantModule(tf.keras.Model):
    """Implements an invariant module performing a permutation-invariant transform.

    For details and rationale, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an invariant module according to [1] which represents a learnable permutation-invariant
        function with an option for learnable pooling.

        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the `tf.keras.Model` constructor.
        """

        super().__init__(**kwargs)

        # Create internal functions
        self.s1 = Sequential([Dense(**settings["dense_s1_args"]) for _ in range(settings["num_dense_s1"])])
        self.s2 = Sequential([Dense(**settings["dense_s2_args"]) for _ in range(settings["num_dense_s2"])])

        # Pick pooling function
        if settings["pooling_fun"] == "mean":
            pooling_fun = partial(tf.reduce_mean, axis=-2)
        elif settings["pooling_fun"] == "max":
            pooling_fun = partial(tf.reduce_max, axis=-2)
        else:
            if callable(settings["pooling_fun"]):
                pooling_fun = settings["pooling_fun"]
            else:
                raise ConfigurationError("pooling_fun argument not understood!")
        self.pooler = pooling_fun

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable invariant transform.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size,..., x_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size,..., out_dim)
        """

        x_reduced = self.pooler(self.s1(x, **kwargs))
        out = self.s2(x_reduced, **kwargs)
        return out


class EquivariantModule(tf.keras.Model):
    """Implements an equivariant module performing an equivariant transform.

    For details and justification, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an equivariant module according to [1] which combines equivariant transforms
        with nested invariant transforms, thereby enabling interactions between set members.

        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model`` constructor.
        """

        super().__init__(**kwargs)

        self.invariant_module = InvariantModule(settings)
        self.s3 = Sequential([Dense(**settings["dense_s3_args"]) for _ in range(settings["num_dense_s3"])])

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable equivariant transform.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, ..., x_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, ..., equiv_dim)
        """

        # Store shape of x, will be (batch_size, ..., some_dim)
        shape = tf.shape(x)

        # Example: Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module(x, **kwargs)
        out_inv = tf.expand_dims(out_inv, -2)
        tiler = [1] * len(shape)
        tiler[-2] = shape[-2]
        out_inv_rep = tf.tile(out_inv, tiler)

        # Concatenate each x with the repeated invariant embedding
        out_c = tf.concat([x, out_inv_rep], axis=-1)

        # Pass through equivariant func
        out = self.s3(out_c, **kwargs)
        return out


class MultiConv1D(tf.keras.Model):
    """Implements an inception-inspired 1D convolutional layer using different kernel sizes."""

    def __init__(self, settings, **kwargs):
        """Creates an inception-like Conv1D layer

        Parameters
        ----------
        settings  : dict
            A dictionary which holds the arguments for the internal ``Conv1D`` layers.
        """

        super().__init__(**kwargs)

        # Create a list of Conv1D layers with different kernel sizes
        # ranging from 'min_kernel_size' to 'max_kernel_size'
        self.convs = [
            Conv1D(kernel_size=f, **settings["layer_args"])
            for f in range(settings["min_kernel_size"], settings["max_kernel_size"])
        ]

        # Create final Conv1D layer for dimensionalitiy reduction
        dim_red_args = {k: v for k, v in settings["layer_args"].items() if k not in ["kernel_size", "strides"]}
        dim_red_args["kernel_size"] = 1
        dim_red_args["strides"] = 1
        self.dim_red = Conv1D(**dim_red_args)

    def call(self, x, **kwargs):
        """Performs a forward pass through the layer.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, n_time_steps, n_time_series)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, n_time_steps, n_filters)
        """

        out = self._multi_conv(x, **kwargs)
        out = self.dim_red(out, **kwargs)
        return out

    def _multi_conv(self, x, **kwargs):
        """Applies the convolutions with different sizes and concatenates outputs."""

        return tf.concat([conv(x, **kwargs) for conv in self.convs], axis=-1)
