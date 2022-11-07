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

from numpy import pi as PI_CONST

from bayesflow import default_settings
from bayesflow.helper_networks import Permutation, ActNorm, DenseCouplingNet
from bayesflow.helper_functions import build_meta_dict
from bayesflow.exceptions import ConfigurationError


class AffineCouplingLayer(tf.keras.Model):
    """Implements a conditional version of the INN coupling layer."""

    def __init__(self, meta):
        """Creates an affine coupling layer, optionally conditional.

        Parameters
        ----------
        meta      : list(dict)
            A list of dictionaries, wherein each dictionary holds parameter-value pairs for a single
            :class:`tf.keras.Dense` layer. All coupling nets are assumed to be equal.
        """

        super().__init__()

        # Coupling net hyperparams
        self.alpha = meta['alpha']
        theta_dim = meta['n_params']
        self.n_out1 = theta_dim // 2
        self.n_out2 = theta_dim // 2 if theta_dim % 2 == 0 else theta_dim // 2 + 1

        # Custom coupling net and settings
        if callable(meta['coupling_design']):
            coupling_type = meta['coupling_design']
            if meta.get('coupling_settings') is None:
                raise ConfigurationError("Need to provide coupling_settings for a custom coupling type.")
            coupling_settings = meta['coupling_settings']

        # String type of dense or attention
        elif type(meta['coupling_design']) is str:
            # Settings type
            if meta.get('coupling_settings') is None:
                user_dict = {}
            elif type(meta.get('coupling_settings')) is dict:
                user_dict = meta.get('coupling_settings')
            else:
                raise ConfigurationError("coupling_settings not understood")

            # Dense
            if meta['coupling_design'] == 'dense':
                coupling_type = DenseCouplingNet
                coupling_settings = build_meta_dict(
                    user_dict=user_dict, default_setting=default_settings.DEFAULT_SETTING_DENSE_COUPLING)
            else:
                raise NotImplementedError('String coupling_design should be one of ["dense"].')
        else:
            raise NotImplementedError('coupling_design argument not understood. Should either be a callable generator or ' +
                                      'a string in ["dense"].')
      
        self.s1 = coupling_type(coupling_settings['s_args'], self.n_out1)
        self.t1 = coupling_type(coupling_settings['t_args'], self.n_out1)
        self.s2 = coupling_type(coupling_settings['s_args'], self.n_out2)
        self.t2 = coupling_type(coupling_settings['t_args'], self.n_out2)

        # Optional permutation
        if meta['use_permutation']:
            self.permutation = Permutation(theta_dim)
        else:
            self.permutation = None

        # Optional activation normalization
        if meta['use_act_norm']:
            self.act_norm = ActNorm(meta)
        else:
            self.act_norm = None

    def call(self, target_or_z, condition, inverse=False, **kwargs):
        """Performs one pass through an invertible chain (either inverse or forward).
        
        Parameters
        ----------
        target_or_z      : tf.Tensor
            the estimation quantites of interest or latent representations z ~ p(z), shape (batch_size, ...)
        condition        : tf.Tensor
            the conditioning data of interest, for instance, x = summary_fun(x), shape (batch_size, ...)
        inverse          : bool, optional, default: False
            Flag indicating whether to run the block forward or backward.
        
        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (batch_size, )

        u               :  tf.Tensor
            If inverse=True: The transformed out, shape (batch_size, inp_dim)

        Important
        ---------
        If ``inverse=False``, the return is ``(v, log_det_J)``.\n
        If ``inverse=True``, the return is ``(z)``
        """
        
        if not inverse:
            return self.forward(target_or_z, condition, **kwargs)
        return self.inverse(target_or_z, condition, **kwargs)

    @tf.function
    def forward(self, target, condition, **kwargs):
        """Performs a forward pass through a coupling layer with an optinal `Permutation` and `ActNorm` layers."""

        # Initialize log_det_Js accumulator
        log_det_Js = tf.zeros(1)
        
        # Normalize activation, if specified
        if self.act_norm is not None:
            target, log_det_J_act = self.act_norm(target)
            log_det_Js += log_det_J_act

        # Permute, if indicated
        if self.permutation is not None:
            target = self.permutation(target)

        # Pass through coupling layer
        z, log_det_J_c = self._forward(target, condition, **kwargs)
        log_det_Js += log_det_J_c

        return z, log_det_Js

    @tf.function
    def inverse(self, z, condition, **kwargs):
        """Performs an inverse pass through a coupling layer with an optinal `Permutation` and `ActNorm` layers."""

        # Pass through coupling layer
        target = self._inverse(z, condition, **kwargs)

        # Pass through optional permutation
        if self.permutation is not None:
            target = self.permutation(target, inverse=True)
        
        # Pass through activation normalization
        if self.act_norm is not None:
            target = self.act_norm(target, inverse=True)
        return target

    @tf.function
    def _forward(self, target, condition, **kwargs):
        """ Performs a forward pass through the coupling block. Used internally by the instance.

        Parameters
        ----------
        target     : tf.Tensor
            the estimation quantities of interest, for instance, parameter vector of shape (batch_size, theta_dim)
        condition  : tf.Tensor or None
            the conditioning vector of interest, for instance, x = summary(x), shape (batch_size, summary_dim)

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (batch_size, )
        """

        # Split parameter vector
        u1, u2 = tf.split(target, [self.n_out1, self.n_out2], axis=-1)

        # Pre-compute network outputs for v1
        s1 = self.s1(u2, condition, **kwargs)
        # Clamp s1 if specified
        if self.alpha is not None:
            s1 = (2. * self.alpha / PI_CONST) * tf.math.atan(s1 / self.alpha)
        t1 = self.t1(u2, condition, **kwargs)
        v1 = u1 * tf.exp(s1) + t1

        # Pre-compute network outputs for v2
        s2 = self.s2(v1, condition, **kwargs)
        # Clamp s2 if specified
        if self.alpha is not None:
            s2 = (2. * self.alpha / PI_CONST) * tf.math.atan(s2 / self.alpha)
        t2 = self.t2(v1, condition, **kwargs)
        v2 = u2 * tf.exp(s2) + t2
        v = tf.concat((v1, v2), axis=-1)

        # Compute ldj, # log|J| = log(prod(diag(J))) -> according to inv architecture
        log_det_J = tf.reduce_sum(s1, axis=-1) + tf.reduce_sum(s2, axis=-1)
        return v, log_det_J 

    @tf.function
    def _inverse(self, z, condition, **kwargs):
        """Performs an inverse pass through the coupling block. Used internally by the instance.

        Parameters
        ----------
        z         : tf.Tensor
            latent variables z ~ p(z), shape (batch_size, theta_dim)
        condition  : tf.Tensor or None
            The conditioning vector of interest, for instance, x = summary(x), shape (batch_size, summary_dim).

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (batch_size, )
        """

        v1, v2 = tf.split(z, [self.n_out1, self.n_out2], axis=-1)

        # Pre-Compute s2
        s2 = self.s2(v1, condition, **kwargs)
        # Clamp s2 if specified
        if self.alpha is not None:
            s2 = (2. * self.alpha / PI_CONST) * tf.math.atan(s2 / self.alpha)
        u2 = (v2 - self.t2(v1, condition, **kwargs)) * tf.exp(-s2)

        # Pre-Compute s1
        s1 = self.s1(u2, condition, **kwargs)
        # Clamp s1 if specified
        if self.alpha is not None:
            s1 = (2. * self.alpha / PI_CONST) * tf.math.atan(s1 / self.alpha)
        u1 = (v1 - self.t1(u2, condition, **kwargs)) * tf.exp(-s1)
        u = tf.concat((u1, u2), axis=-1)

        return u
