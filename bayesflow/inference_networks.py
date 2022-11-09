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

from bayesflow import default_settings
from bayesflow.coupling_networks import AffineCouplingLayer
from bayesflow.helper_functions import build_meta_dict


class InvertibleNetwork(tf.keras.Model):
    """Implements a chain of conditional invertible coupling layers for conditional density estimation."""

    def __init__(self, meta={}):
        """Creates a chain of coupling layers with optional `ActNorm` layers in-between.

        Parameters
        ----------
        meta : dict
            The configuration settings for the invertible network.
        """

        super().__init__()

        # Create settings dictionary
        meta = build_meta_dict(user_dict=meta,
                               default_setting=default_settings.DEFAULT_SETTING_INVERTIBLE_NET)

        # Create sequence of coupling layers and store reference to dimensionality
        self.coupling_layers = [AffineCouplingLayer(meta) for _ in range(meta['n_coupling_layers'])]
        self.soft_flow = meta['use_soft_flow']
        self.soft_low = meta['soft_flow_bounds'][0]
        self.soft_high = meta['soft_flow_bounds'][1]
        self.latent_dim = meta['n_params']

    def call(self, targets, condition, inverse=False, **kwargs):
        """Performs one pass through an invertible chain (either inverse or forward).

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

        # Add noise to target if using SoftFlow, use explicitly
        # not in call(), since methods are public
        if self.soft_flow and condition is not None:
            # Needs to be concatinable with condition
            shape_scale = (condition.shape[0], 1) if len(condition.shape) == 2 else (condition.shape[0], condition.shape[1], 1)
            # Case training mode
            if kwargs.get('training'):
                noise_scale = tf.random.uniform(shape=shape_scale, minval=self.soft_low, maxval=self.soft_high)
            # Case inference mode
            else:
                noise_scale = tf.zeros(shape=shape_scale) + self.soft_low
            # Perturb data with noise (will broadcast to all dimensions)
            if len(shape_scale) == 2 and len(targets.shape) == 3:
                targets += tf.expand_dims(noise_scale, axis=1) * tf.random.normal(shape=targets.shape)
            else:
                targets += noise_scale * tf.random.normal(shape=targets.shape)
            # Augment condition with noise scale variate
            condition = tf.concat((condition, noise_scale), axis=-1)

        z = targets
        log_det_Js = []
        for layer in self.coupling_layers:
            z, log_det_J = layer(z, condition, **kwargs)
            log_det_Js.append(log_det_J)

        # Sum Jacobian determinants for all layers (coupling blocks) to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        return z, log_det_J

    @tf.function
    def inverse(self, z, condition, **kwargs):
        """Performs a reverse pass through the chain. Assumes only used
        in inference mode, so **kwargs contains `training=False`."""

        # Add noise to target if using SoftFlow, use explicitly
        # not in call(), since methods are public
        if self.soft_flow and condition is not None:

            # Needs to be concatinable with condition
            shape_scale = (condition.shape[0], 1) if len(condition.shape) == 2 else (condition.shape[0], condition.shape[1], 1)
            noise_scale = tf.zeros(shape=shape_scale) + 2.*self.soft_low

            # Augment condition with noise scale variate
            condition = tf.concat((condition, noise_scale), axis=-1)

        target = z
        for layer in reversed(self.coupling_layers):
            target = layer(target, condition, inverse=True, **kwargs)
        return target


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

        super().__init__()

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
