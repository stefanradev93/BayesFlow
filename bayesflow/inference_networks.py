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
from bayesflow.coupling_networks import CouplingLayer
from bayesflow.helper_functions import build_meta_dict
from bayesflow.helper_networks import MCDropout


class InvertibleNetwork(tf.keras.Model):
    """Implements a chain of conditional invertible coupling layers for conditional density estimation."""

    available_designs = ("affine", "spline", "interleaved")

    def __init__(
        self,
        num_params,
        num_coupling_layers=6,
        coupling_design="affine",
        coupling_settings=None,
        permutation="fixed",
        use_act_norm=True,
        act_norm_init=None,
        use_soft_flow=False,
        soft_flow_bounds=(1e-3, 5e-2),
        **kwargs,
    ):
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

        [4] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
        Neural spline flows. Advances in Neural Information Processing Systems, 32.

        [5] Kingma, D. P., & Dhariwal, P. (2018).
        Glow: Generative flow with invertible 1x1 convolutions.
        Advances in Neural Information Processing Systems, 31.

        Parameters
        ----------
        num_params            : int
            The number of parameters to perform inference on. Equivalently, the dimensionality of the
            latent space.
        num_coupling_layers   : int, optional, default: 6
            The number of coupling layers to use as defined in [1] and [2]. In general, more coupling layers
            will give you more expressive power, but will be slower and may need more simulations to train.
            Typically, between 4 and 10 coupling layers should suffice for most applications.
        coupling_design       : str or callable, optional, default: 'affine'
            The type of internal coupling network to use. Must be in ['affine', 'spline', 'interleaved'].
            The first corresponds to the architecture in [3, 5], the second corresponds to a modified
            version of [4]. The third option will alternate between affine and spline layers, for example,
            if num_coupling_layers == 3, the chain will consist of ["affine", "spline", "affine"] layers.

            In general, spline couplings run slower than affine couplings, but require fewer coupling
            layers. Spline couplings may work best with complex (e.g., multimodal) low-dimensional
            problems. The difference will become less and less pronounced as we move to higher dimensions.

            Note: This is the first setting you may want to change, if inference does not work as expected!
        coupling_settings     : dict or None, optional, default: None
            The coupling network settings to pass to the internal coupling layers. See ``default_settings``
            for possible settings. Below are two examples.

            Examples:

            1. If using ``coupling_design='affine``, you may want to turn on Monte Carlo Dropout and
            use an ELU activation function for the internal networks. You can do this by providing:
            ``
            coupling_settings={
                'mc_dropout' : True,
                'dense_args' : dict(units=128, activation='elu')
            }
            ``

            2. If using ``coupling_design='spline'``, you may want to change the number of learnable bins
            and increase the dropout probability (i.e., more regularization to guard against overfitting):
            ``
            coupling_settings={
                'dropout_prob': 0.2,
                'bins' : 32,
            }
            ``
        permutation           : str or None, optional, default: 'fixed'
            Whether to use permutations between coupling layers. Highly recommended if ``num_coupling_layers > 1``
            Important: Must be in ['fixed', 'learnable', None]
        use_act_norm          : bool, optional, default: True
            Whether to use activation normalization after each coupling layer, as used in [5].
            Recommended to keep default.
        act_norm_init         : np.ndarray of shape (num_simulations, num_params) or None, optional, default: None
            Optional data-dependent initialization for the internal ``ActNorm`` layers, as done in [5]. Could be helpful
            for deep invertible networks.
        use_soft_flow         : bool, optional, default: False
            Whether to perturb the target distribution (i.e., parameters) with small amount of independent
            noise, as done in [2]. Could be helpful for degenerate distributions.
        soft_flow_bounds      : tuple(float, float), optional, default: (1e-3, 5e-2)
            The bounds of the continuous uniform distribution from which the noise scale would be sampled
            at each iteration. Only relevant when ``use_soft_flow=True``.
        **kwargs              : dict
            Optional keyword arguments (e.g., name) passed to the tf.keras.Model __init__ method.
        """

        super().__init__(**kwargs)

        layer_settings = dict(
            latent_dim=num_params,
            permutation=permutation,
            use_act_norm=use_act_norm,
            act_norm_init=act_norm_init,
        )
        self.coupling_layers = self._create_coupling_layers(
            layer_settings, coupling_settings, coupling_design, num_coupling_layers
        )
        self.soft_flow = use_soft_flow
        self.soft_low = soft_flow_bounds[0]
        self.soft_high = soft_flow_bounds[1]
        self.permutation = permutation
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

        Notes
        -----
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``.
        """

        if inverse:
            return self.inverse(targets, condition, **kwargs)
        return self.forward(targets, condition, **kwargs)

    def forward(self, targets, condition, **kwargs):
        """Performs a forward pass through the chain."""

        # Add noise to target if using SoftFlow, use explicitly
        # not in call(), since methods are public
        if self.soft_flow and condition is not None:
            # Extract shapes of tensors
            target_shape = tf.shape(targets)
            condition_shape = tf.shape(condition)

            # Needs to be concatinable with condition
            if len(condition_shape) == 2:
                shape_scale = (condition_shape[0], 1)
            else:
                shape_scale = (condition_shape[0], condition_shape[1], 1)

            # Case training mode
            if kwargs.get("training"):
                noise_scale = tf.random.uniform(shape=shape_scale, minval=self.soft_low, maxval=self.soft_high)
            # Case inference mode
            else:
                noise_scale = tf.zeros(shape=shape_scale) + self.soft_low

            # Perturb data with noise (will broadcast to all dimensions)
            if len(shape_scale) == 2 and len(target_shape) == 3:
                targets += tf.expand_dims(noise_scale, axis=1) * tf.random.normal(shape=target_shape)
            else:
                targets += noise_scale * tf.random.normal(shape=target_shape)

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

    def inverse(self, z, condition, **kwargs):
        """Performs a reverse pass through the chain. Assumes that it is only used
        in inference mode, so ``**kwargs`` contains ``training=False``."""

        # Add noise to target if using SoftFlow, use explicitly
        # not in call(), since methods are public
        if self.soft_flow and condition is not None:
            # Needs to be concatinable with condition
            shape_scale = (
                (condition.shape[0], 1) if len(condition.shape) == 2 else (condition.shape[0], condition.shape[1], 1)
            )
            noise_scale = tf.zeros(shape=shape_scale) + 2.0 * self.soft_low

            # Augment condition with noise scale variate
            condition = tf.concat((condition, noise_scale), axis=-1)

        target = z
        for layer in reversed(self.coupling_layers):
            target = layer(target, condition, inverse=True, **kwargs)
        return target

    @staticmethod
    def _create_coupling_layers(settings, coupling_settings, coupling_design, num_coupling_layers):
        """Helper method to create a list of coupling layers. Takes care
        of the different options for coupling design.
        """

        if coupling_design not in InvertibleNetwork.available_designs:
            raise NotImplementedError("Coupling design should be one of", InvertibleNetwork.available_designs)

        # Case affine or spline
        if coupling_design != "interleaved":
            design = coupling_design
            _coupling_settings = coupling_settings
            coupling_layers = [
                CouplingLayer(coupling_design=design, coupling_settings=_coupling_settings, **settings)
                for _ in range(num_coupling_layers)
            ]
        # Case interleaved, starts with affine
        else:
            coupling_layers = []
            designs = (["affine", "spline"] * int(np.ceil(num_coupling_layers / 2)))[:num_coupling_layers]
            for design in designs:
                # Fail gently, if neither None, nor a dictionary with keys ("spline", "affine")
                _coupling_settings = None if coupling_settings is None else coupling_settings[design]
                layer = CouplingLayer(coupling_design=design, coupling_settings=_coupling_settings, **settings)
                coupling_layers.append(layer)
        return coupling_layers

    @classmethod
    def create_config(cls, **kwargs):
        """ "Used to create the settings dictionary for the internal networks of the invertible
        network. Will fill in missing"""

        settings = build_meta_dict(user_dict=kwargs, default_setting=default_settings.DEFAULT_SETTING_INVERTIBLE_NET)
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

    def __init__(self, num_models, dense_args=None, num_dense=3, output_activation="softplus", **kwargs):
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
        self.dense = tf.keras.Sequential([tf.keras.layers.Dense(**dense_args) for _ in range(num_dense)])

        # The layer to output model evidences
        self.alpha_layer = tf.keras.layers.Dense(
            num_models,
            activation=output_activation,
            **{k: v for k, v in dense_args.items() if k != "units" and k != "activation"},
        )

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
        evidence = alpha + 1.0
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
            [np.default_rng().dirichlet(alpha[n, :], size=n_samples) for n in range(n_datasets)], axis=1
        )
        return pm_samples

    @classmethod
    def create_config(cls, **kwargs):
        """ "Used to create the settings dictionary for the internal networks of the invertible
        network. Will fill in missing"""

        settings = build_meta_dict(user_dict=kwargs, default_setting=default_settings.DEFAULT_SETTING_EVIDENTIAL_NET)
        return settings


class PMPNetwork(tf.keras.Model):
    """Implements a network that approximates posterior model probabilities (PMPs) as employed in [1].

    [1] Elsemüller, L., Schnuerch, M., Bürkner, P. C., & Radev, S. T. (2023).
        A Deep Learning Method for Comparing Bayesian Hierarchical Models.
        arXiv preprint arXiv:2301.11873.
    """

    def __init__(
        self,
        num_models,
        dense_args=None,
        num_dense=3,
        dropout=True,
        mc_dropout=False,
        dropout_prob=0.05,
        output_activation=tf.nn.softmax,
        **kwargs,
    ):
        """Creates an instance of a PMP network for amortized model comparison.

        Parameters
        ----------
        num_models        : int
            The number of candidate (competing models) for the comparison scenario.
        dense_args        : dict or None, optional, default: None
            The arguments for a tf.keras.layers.Dense layer. If None, defaults will be used.
        num_dense         : int, optional, default: 3
            The number of dense layers for the main network part.
        dropout           : bool, optional, default: True
            Whether to use dropout in-between the hidden layers.
        mc_dropout        : bool, optional, default: False
            Whether to use dropout Monte Carlo dropout (i.e., Bayesian approximation) during inference
        dropout_prob      : float in (0, 1), optional, default: 0.05
            The dropout probability. Only has effecft if ``dropout=True`` or ``mc_dropout=True``
        output_activation : callable, optional, default: tf.nn.softmax
            The activation function to apply to the network outputs.
            Important: Needs to have positive outputs and be bounded between 0 and 1.
        **kwargs          : dict, optional, default: {}
            Optional keyword arguments (e.g., name) passed to the ``tf.keras.Model`` __init__ method.
        """

        super().__init__(**kwargs)

        # Pick default settings, if None provided
        if dense_args is None:
            dense_args = default_settings.DEFAULT_SETTING_DENSE_PMP

        # Sequential model with optional (MC) Dropout
        self.net = tf.keras.Sequential()
        for _ in range(num_dense):
            self.net.add(tf.keras.layers.Dense(**dense_args))
            if mc_dropout:
                self.net.add(MCDropout(dropout_prob))
            elif dropout:
                self.net.add(tf.keras.layers.Dropout(dropout_prob))
            else:
                pass
        self.output_layer = tf.keras.layers.Dense(num_models)
        self.output_activation = output_activation
        self.num_models = num_models

    def call(self, condition, return_probs=True, **kwargs):
        """Forward pass through the network. Computes approximated PMPs given a batch of data
        and optional concatenated context, typically passed through a summary network.

        Parameters
        ----------
        condition    : tf.Tensor of shape (batch_size, ...)
            The input variables used for determining ``p(model | condition)``
        return_probs : bool, optional, default: True
            Whether to return probabilities or logits (pre-activation, unnormalized)

        Returns
        -------
        out          : tf.Tensor of shape (batch_size, ..., num_models)
            The approximated PMPs (post-activation) or logits (pre-activation)
        """

        rep = self.net(condition, **kwargs)
        logits = self.output_layer(rep, **kwargs)
        if return_probs:
            return self.output_activation(logits)
        return logits

    def posterior_probs(self, condition, **kwargs):
        """Shortcut function to obtain posterior probabilities given a
        condition tensor (e.g., summary statistics of data sets).

        Parameters
        ----------
        condition : tf.Tensor of shape (batch_size, ...)
            The input variables used for determining ``p(model | condition)``

        Returns
        -------
        out       : tf.Tensor of shape (batch_size, ..., num_models)
            The approximated PMPs
        """

        return self(condition, return_probs=True, **kwargs)

    def logits(self, condition, **kwargs):
        """Shortcut function to obtain logits given a condition tensor
        (e.g., summary statistics of data sets).

        Parameters
        ----------
        condition : tf.Tensor of shape (batch_size, ...)
            The input variables used for determining ``p(model | condition)``

        Returns
        -------
        out       : tf.Tensor of shape (batch_size, ..., num_models)
            The approximated PMPs
        """

        return self(condition, return_probs=False, **kwargs)

    @classmethod
    def create_config(cls, **kwargs):
        """Used to create the settings dictionary for the internal networks of the
        network. Will fill in missing."""

        settings = build_meta_dict(user_dict=kwargs, default_setting=default_settings.DEFAULT_SETTING_PMP_NET)
        return settings
