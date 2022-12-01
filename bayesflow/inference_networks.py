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

    def __init__(self, num_params, num_coupling_layers=4, coupling_net_settings=None, 
                 coupling_design='dense', soft_clamping=1.9, use_permutation=True, use_act_norm=True, 
                 act_norm_init=None, use_soft_flow=False, soft_flow_bounds=(1e-3, 5e-2), **kwargs):
        """Creates a chain of coupling layers with optional `ActNorm` layers in-between. Implements ideas from:

        [1] Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020). 
        BayesFlow: Learning complex stochastic models with invertible neural networks. 
        IEEE Transactions on Neural Networks and Learning Systems.

        [2] Kim, H., Lee, H., Kang, W. H., Lee, J. Y., & Kim, N. S. (2020). 
        Softflow: Probabilistic framework for normalizing flow on manifolds. 
        Advances in Neural Information Processing Systems, 33, 16388-16397.

        [3] Ardizzone, L., Kruse, J., Lüth, C., Bracher, N., Rother, C., & Köthe, U. (2020). 
        Conditional invertible neural networks for diverse image-to-image translation. 
        In DAGM German Conference on Pattern Recognition (pp. 373-387). Springer, Cham.

        [4] Kingma, D. P., & Dhariwal, P. (2018). 
        Glow: Generative flow with invertible 1x1 convolutions. 
        Advances in Neural Information Processing Systems, 31.

        Parameters
        ----------
        num_params            : int
            The number of parameters to perform inference on. Equivalently, the dimensionality of the
            latent space. 
        num_coupling_layers   : int, optional, default: 4
            The number of coupling layers to use as defined in [1] and [2]. In general, more coupling layers
            will give you more expressive power, but will be slower and may need more simulations to train.
            Typically, between 4 and 10 coupling layers should suffice for most applications.
        coupling_net_settings : dict or None, optional, default: None
            The coupling network settings to pass to the internal coupling layers. See `default_settings`
            for the required entries.
        coupling_design       : str or callable, optional, default: 'dense'
            The type of internal coupling network to use. Currently, only 'dense' is understood as a
            string argument, but you can also pass a callable which constructs a custom network. In that case,
            the `coupling_net_settings` will be passed as a first argument to the callable.
        soft_clamping         : float, optional, default: 1.9
            The soft clamping parameter `alpha` in [3]. Typically you would not touch this.
        use_permutation       : bool, optional, default: True
            Whether to use fixed permutations between coupling layers. Highly recommended.
        use_act_norm          : bool, optional, default: True
            Whether to use activation normalization after each coupling layer, as used in [4].
            Recommended to keep default.
        act_norm_init         : np.ndarray of shape (num_simulations, num_params) or None, optional, default: None
            Optional data-dependent initialization for the internal `ActNorm` layers, as done in [4]. Could be helpful 
            for deep invertible networks.
        use_soft_flow         : bool, optional, default: False
            Whether to perturb the taregt distribution (i.e., parameters) with small amount of independent
            noise, as done in [3]. Could be helpful for degenrate distributions.
        soft_flow_bounds      : tuple(float, float), optional, default: (1e-3, 5e-2)
            The bounds of the continuous uniform distribution from which the noise scale would be sampled
            at each iteration. Only relevant when `use_soft_flow=True`.
        **kwargs              : dict
            Optional keyword arguments (e.g., name) passed to the tf.keras.Model __init__ method.
        """

        super().__init__(**kwargs)

        # Create settings dict for coupling layer
        settings = dict(
            latent_dim=num_params,
            coupling_net_settings=coupling_net_settings,
            coupling_design=coupling_design,
            use_permutation=use_permutation,
            use_act_norm=use_act_norm,
            act_norm_init=act_norm_init,
            alpha=soft_clamping
        )

        # Create sequence of coupling layers and store reference to dimensionality
        self.coupling_layers = [AffineCouplingLayer(settings) for _ in range(num_coupling_layers)]

        # Store attributes
        self.soft_flow = use_soft_flow
        self.soft_low = soft_flow_bounds[0]
        self.soft_high = soft_flow_bounds[1]
        self.use_permutation = use_permutation
        self.use_act_norm = use_act_norm 
        self.latent_dim = num_params

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
        """Performs a reverse pass through the chain. Assumes that it is only used
        in inference mode, so ``**kwargs`` contains ``training=False``."""

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

    @classmethod
    def create_config(cls, **kwargs):
        """"Used to create the settings dictionary for the internal networks of the invertible
        network. Will fill in missing """

        settings = build_meta_dict(user_dict=kwargs,
                                   default_setting=default_settings.DEFAULT_SETTING_INVERTIBLE_NET)
        return settings


class EvidentialNetwork(tf.keras.Model):
    """Implements a network whose outputs are the concentration parameters of a Dirichlet density.

    Follows ideas from:
    
    [1] Radev, S. T., D'Alessandro, M., Mertens, U. K., Voss, A., Köthe, U., & Bürkner, P. C. (2021). 
    Amortized Bayesian model comparison with evidential deep learning. 
    IEEE Transactions on Neural Networks and Learning Systems.

    [2] Sensoy, M., Kaplan, L., & Kandemir, M. (2018). 
    Evidential deep learning to quantify classification uncertainty. 
    Advances in neural information processing systems, 31.
    """

    def __init__(self, num_models, dense_args=None, num_dense=3, output_activation='softplus', **kwargs):
        """Creates an instance of an evidential network for amortized model comparison.

        Parameters
        ----------
        num_models        : int
            The number of candidate (competing models) for the comparison scenario.
        dense_args        : dict or None, optional, default: None
            The arguments for a tf.keras.layers.Dense layer. If None, defaults will be used.
        num_dense         : int, optional, default: 3
            The number of dense layers for the main network part. 
        output_activation : str or callable, optional, default: 'softplus'
            The activation function to use for the network outputs. 
            Important: needs to have positive outputs.
        **kwargs          : dict, optional, default: {}
            Optional keyword arguments (e.g., name) passed to the tf.keras.Model __init__ method.
        """

        super().__init__(**kwargs)

        if dense_args is None:
            dense_args = default_settings.DEFAULT_SETTING_DENSE_EVIDENTIAL

        # A network to increase representation power
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(**dense_args)
            for _ in range(num_dense)
        ])

        # The layer to output model evidences
        self.alpha_layer = tf.keras.layers.Dense(
            num_models, activation=output_activation, 
            **{k: v for k, v in dense_args.items() if k != 'units' and k != 'activation'})

        self.num_models = num_models

    def call(self, condition, **kwargs):
        """Computes evidences for model comparison given a batch of data and optional concatenated context, 
        typically passed through a summayr network.

        Parameters
        ----------
        condition  : tf.Tensor of shape (batch_size, ...)
            The input variables used for determining ``p(model | condition)``

        Returns
        -------
        evidence    : tf.Tensor of shape (batch_size, num_models) -- the learned model evidences
        """

        return self.evidence(condition, **kwargs)

    @tf.function
    def evidence(self, condition, **kwargs):
        rep = self.dense(condition, **kwargs)
        alpha = self.alpha_layer(rep, **kwargs)
        evidence = alpha + 1.
        return evidence

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
            The posterior draws from the Dirichlet distribution, shape (num_samples, num_batch, num_models)
        """

        alpha = self.evidence(condition, **kwargs)
        n_datasets = alpha.shape[0]
        pm_samples = np.stack(
            [np.default_rng().dirichlet(alpha[n, :], size=n_samples) for n in range(n_datasets)], axis=1)
        return pm_samples

    @classmethod
    def create_config(cls, **kwargs):
        """"Used to create the settings dictionary for the internal networks of the invertible
        network. Will fill in missing """

        settings = build_meta_dict(user_dict=kwargs,
                                   default_setting=default_settings.DEFAULT_SETTING_EVIDENTIAL_NET)
        return settings
