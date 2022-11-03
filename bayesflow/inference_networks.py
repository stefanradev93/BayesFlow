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
from scipy.stats import multivariate_t, multivariate_normal

import tensorflow as tf

from bayesflow import default_settings
from bayesflow.helper_networks import Permutation, TailNetwork, ActNorm, DenseCouplingNet
from bayesflow.helper_functions import build_meta_dict
from bayesflow.exceptions import ConfigurationError


class ConditionalCouplingLayer(tf.keras.Model):
    """Implements a conditional version of the INN coupling layer."""

    def __init__(self, meta):
        """Creates a conditional invertible block.

        Parameters
        ----------
        meta      : list(dict)
            A list of dictionaries, wherein each dictionary holds parameter-value pairs for a single
            :class:`tf.keras.Dense` layer. All coupling nets are assumed to be equal.
        """

        super(ConditionalCouplingLayer, self).__init__()

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
            s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
        t1 = self.t1(u2, condition, **kwargs)
        v1 = u1 * tf.exp(s1) + t1

        # Pre-compute network outputs for v2
        s2 = self.s2(v1, condition, **kwargs)
        # Clamp s2 if specified
        if self.alpha is not None:
            s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
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
            s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
        u2 = (v2 - self.t2(v1, condition, **kwargs)) * tf.exp(-s2)

        # Pre-Compute s1
        s1 = self.s1(u2, condition, **kwargs)
        # Clamp s1 if specified
        if self.alpha is not None:
            s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
        u1 = (v1 - self.t1(u2, condition, **kwargs)) * tf.exp(-s1)
        u = tf.concat((u1, u2), axis=-1)

        return u


class InvertibleNetwork(tf.keras.Model):
    """Implements a chain of conditional invertible coupling layers for conditional density estimation."""

    def __init__(self, meta={}):
        """Creates a chain of coupling layers with optional `ActNorm` layers in-between.

        Parameters
        ----------
        meta : list(dict)
            A list of dictionaries, where each dictionary holds parameter-value pairs
            for a single :class:`keras.Dense` layer

        Notes
        -----
        Important: Currently supports Gaussiand and Student-t latent spaces only.
        """

        super(InvertibleNetwork, self).__init__()

        # Create settings dictionary
        meta = build_meta_dict(user_dict=meta,
                               default_setting=default_settings.DEFAULT_SETTING_INVERTIBLE_NET)

        # Create sequence of coupling layers
        self.coupling_layers = [ConditionalCouplingLayer(meta) for _ in range(meta['n_coupling_layers'])]

        # Determine tail network 
        if meta.get('tail_network') is None:
            self.tail_network = None
        elif meta.get('tail_network') is True:
            self.tail_network = TailNetwork(default_settings.DEFAULT_SETTING_TAIL_NET.meta_dict)
        elif type(meta.get('tail_network')) is dict:
            tail_meta = build_meta_dict(user_dict=meta.get('tail_network'),
                               default_setting=default_settings.DEFAULT_SETTING_TAIL_NET)
            self.tail_network = TailNetwork(tail_meta)
        elif callable(meta.get('tail_network')):
            self.tail_network = meta.get('tail_network')
        else:
            raise ConfigurationError("tail_network argument type should be one of (True, None, dict, callable)")
            
        self.z_dim = meta['n_params']

    def call(self, targets, condition, inverse=False, **kwargs):
        """ Performs one pass through an invertible chain (either inverse or forward).

        Parameters
        ----------
        targets   : tf.Tensor
            The estimation quantities of interest, shape (batch_size, ...)
        condition : tf.Tensor
            The conditional data x, shape (batch_size, summary_dim)
        inverse   : bool, default: False
            Flag indicating whether to run the chain forward or backwards

        Returns
        -------
        (z, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, ...), log_det_J shape: (batch_size, ...)

        target          :  tf.Tensor
            If inverse=True: The transformed out, shape (batch_size, ...)

        Important
        ---------
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``.
        """
        
        if inverse:
            return self.inverse(targets, condition, **kwargs)
        return self.forward(targets, condition, **kwargs)

    @tf.function
    def forward(self, targets, condition, **kwargs):
        """Performs a forward pass though the chain."""

        z = targets
        log_det_Js = []
        for layer in self.coupling_layers:
            z, log_det_J = layer(z, condition, **kwargs)
            log_det_Js.append(log_det_J)
        # Sum Jacobian determinants for all layers (coupling blocks) to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        
        # Adaptive tails or simply Gaussian
        if self.tail_network is not None:
            v = self.tail_network(condition, **kwargs)
            return v, z, log_det_J
        else:
            return z, log_det_J

    @tf.function
    def inverse(self, z, condition, **kwargs):
        """ Performs a reverse pass through the chain."""

        target = z
        for layer in reversed(self.coupling_layers):
            target = layer(target, condition, inverse=True, **kwargs)
        return target

    def sample(self, condition, n_samples, **kwargs):
        """ Samples from the inverse model given a single data instance or a batch of data instances.

        Parameters
        ----------
        condition : tf.Tensor
            The conditioning data set(s) of interest, shape (n_data_sets, summary_dim)
        n_samples : int
            Number of samples to obtain from the approximate posterior
        Returns
        -------
        theta_samples : tf.Tensor or np.array
            Parameter samples, shape (n_samples, n_datasets, n_params)
        """

        # Handle unconditional case
        if condition is None:
            z_samples = tf.random.normal(shape=(n_samples, self.z_dim))
            target_samples = self.inverse(z_samples, condition, **kwargs)
            return target_samples

        # Sample from a unit Gaussian
        if self.tail_network is None:
            z_samples = tf.random.normal(shape=(int(condition.shape[0]), n_samples, self.z_dim))
        # Sample from a t-distro    
        else:
            dfs = self.tail_network(condition, **kwargs).numpy().squeeze(axis=-1)
            loc = np.zeros(self.z_dim)
            shape = np.eye(self.z_dim)
            z_samples = tf.stack(
                [multivariate_t(df=df, loc=loc, shape=shape).rvs(n_samples) 
                for df in dfs]
            )
        
        # Inverse pass
        target_samples = self.inverse(z_samples, condition, **kwargs)

        # Remove extra batch-dimension, if single instance
        if int(target_samples.shape[0]) == 1:
            target_samples = target_samples[0]
        return target_samples

    def log_density(self, targets, condition, **kwargs):
        """ Calculates the approximate log-density of targets given conditional variables.

        Parameters
        ----------
        targets   : tf.Tensor or np.ndarray
            The estimation quantities of interest, expected shape (batch_size, ...)
        condition : tf.Tensor or np.ndarray
            The conditioning variables, expected shape (batch_size, ...)

        Returns
        -------
        loglik    : tf.Tensor of shape (batch_size, ...)
            the approximate log-likelihood of each data point in each data set
        """

        if not self.tail_network:
            z, log_det_J = self.forward(targets, condition, **kwargs)
            k = z.shape[-1]
            log_z_unnorm = -0.5 * tf.math.square(tf.norm(z, axis=-1)) 
            log_z = log_z_unnorm - tf.math.log(tf.math.sqrt((2*np.pi)**k))
            log_pdf = log_z + log_det_J
        else:
            log_pdf = self._log_density_student_t(targets, condition, **kwargs)
        return log_pdf

    def _log_density_gaussian(self, targets, condition, **kwargs):
        """ Calculates the approximate log-density of targets given conditional variables and
        a latent Gaussian distribution.

        Parameters
        ----------
        targets   : tf.Tensor or np.ndarray
            The estimation quantities of interest, expected shape (batch_size, ...)
        condition : tf.Tensor or np.ndarray
            The conditioning variables, expected shape (batch_size, ...)

        Returns
        -------
        loglik    : tf.Tensor of shape (batch_size, ...)
            the approximate log-likelihood of each data point in each data set
        """

        z, log_det_J = self.forward(targets, condition, **kwargs)
        log_z = multivariate_normal(mean=np.zeros(self.z_dim), cov=1.).logpdf(z.numpy())
        log_z = tf.convert_to_tensor(log_z, dtype=log_det_J.dtype)
        log_pdf = log_z + log_det_J
        return log_pdf
    
    def _log_density_student_t(self, targets, condition, **kwargs):
        """ Calculates the approximate log-density of targets given conditional variables and
        a latent student_t distribution

        Parameters
        ----------
        targets   : tf.Tensor or np.ndarray
            The estimation quantities of interest, expected shape (batch_size, ...)
        condition : tf.Tensor or np.ndarray
            The conditioning variables, expected shape (batch_size, ...)

        Returns
        -------
        log_lik   : tf.Tensor of shape (batch_size, ...)
            the approximate log-likelihood of each data point in each data set
        """

        v, z, log_det_J = self.forward(targets, condition, **kwargs)
        batch_size = v.shape[0]
        log_z = [multivariate_t(df=v[b].numpy().item(), loc=np.zeros(self.z_dim), shape=1.).logpdf(z[b].numpy())
                for b in range(batch_size)]
        log_z = tf.stack(log_z, axis=0)
        log_z = tf.cast(log_z, dtype=log_det_J.dtype)
        log_pdf = log_z + log_det_J
        return log_pdf


class EvidentialNetwork(tf.keras.Model):
    """Implements a network whose outputs are the concentration parameters of a Dirichlet density.
    
    Follows the implementation from:
    https://arxiv.org/abs/2004.10629
    """

    def __init__(self, meta={}):
        """Creates an instance of an evidential network for amortized model comparison.

        Parameters
        ----------
        meta  : dict
            A list of dictionaries, where each dictionary holds parameter-value pairs
            for a single :class:`tf.keras.Dense` layer
        """

        super(EvidentialNetwork, self).__init__()

        # Create settings dictionary
        meta = build_meta_dict(user_dict=meta,
                               default_setting=default_settings.DEFAULT_SETTING_EVIDENTIAL_NET)

        # A network to increase representation power
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['dense_args'])
            for _ in range(meta['n_dense'])
        ])

        # The layer to output model evidences
        self.evidence_layer = tf.keras.layers.Dense(
            meta['n_models'], activation=meta['output_activation'], 
            **{k: v for k, v in meta['dense_args'].items() if k != 'units' and k != 'activation'})

        self.n_models = meta['n_models']

    def call(self, condition, **kwargs):
        """Computes evidences for model comparison given a batch of data and optional concatenated context, 
        typically passed through a summayr network.

        Parameters
        ----------
        condition  : tf.Tensor of shape (batch_size, ...)
            The input variables used for determining `p(model | condition)`

        Returns
        -------
        alpha      : tf.Tensor of shape (batch_size, n_models) -- the learned model evidences
        """

        rep = self.dense(condition, **kwargs)
        evidence = self.evidence_layer(rep, **kwargs)
        alpha = evidence + 1
        return alpha

    def sample(self, condition, n_samples, **kwargs):
        """Samples posterior model probabilities from the higher-order Dirichlet density.

        Parameters
        ----------
        condition  : tf.Tensor
            The summary of the observed (or simulated) data, shape (n_data_sets, ...)
        n_samples  : int
            Number of samples to obtain from the approximate posterior
            
        Returns
        -------
        pm_samples : tf.Tensor or np.array
            The posterior draws from the Dirichlet distribution, shape (n_samples, n_batch, n_models)
        """

        # Compute evidential values
        alpha = self(condition, **kwargs)
        n_datasets = alpha.shape[0]

        # Sample for each dataset
        pm_samples = np.stack([np.random.dirichlet(alpha[n, :], size=n_samples) for n in range(n_datasets)], axis=1)
        return pm_samples