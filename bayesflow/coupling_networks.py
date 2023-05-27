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
from numpy import e as EULER_CONST
from numpy import pi as PI_CONST

from bayesflow import default_settings
from bayesflow.exceptions import ConfigurationError
from bayesflow.helper_functions import build_meta_dict
from bayesflow.helper_networks import ActNorm, DenseCouplingNet, Orthogonal, Permutation


class AffineCoupling(tf.keras.Model):
    """Implements a conditional affine coupling block according to [1, 2], with additional
    options, such as residual blocks or Monte Carlo Dropout.


    [1] Kingma, D. P., & Dhariwal, P. (2018).
    Glow: Generative flow with invertible 1x1 convolutions.
    Advances in neural information processing systems, 31.

    [2] Ardizzone, L., Lüth, C., Kruse, J., Rother, C., & Köthe, U. (2019).
    Guided image generation with conditional invertible neural networks.
    arXiv preprint arXiv:1907.02392.
    """

    def __init__(self, dim_out, settings_dict, **kwargs):
        """Creates one half of an affine coupling layer to be used as part of a ``CouplingLayer`` in
        an ``InvertibleNetwork`` instance.

        Parameters
        ----------
        dim_out       : int
            The output dimensionality of the affine coupling layer.
        settings_dict : dict
            The settings for the inner networks. Defaults will use:
            ``settings_dict={
                "dense_args"    : dict(units=128, activation="relu"),
                "num_dense"     : 2,
                "spec_norm"     : False,
                "mc_dropout"    : False,
                "dropout"       : True,
                "residual"      : False,
                "dropout_prob"  : 0.01,
                "soft_clamping" : 1.9
            }
            ``
        """
        super().__init__(**kwargs)

        self.dim_out = dim_out
        self.soft_clamp = settings_dict["soft_clamping"]

        # Check if separate settings for s and t are provided and adjust accordingly
        if settings_dict.get("s_args") is not None and settings_dict.get("t_args") is not None:
            s_settings, t_settings = settings_dict.get("s_args"), settings_dict.get("t_args")
        elif settings_dict.get("s_args") is not None and settings_dict.get("t_args") is None:
            raise ConfigurationError("s_args were provided, but you also need to provide t_args!")
        elif settings_dict.get("s_args") is None and settings_dict.get("t_args") is not None:
            raise ConfigurationError("t_args were provided, but you also need to provide s_args!")
        else:
            s_settings, t_settings = settings_dict, settings_dict

        # Internal network (learnable scale and translation)
        self.scale = DenseCouplingNet(s_settings, dim_out)
        self.translate = DenseCouplingNet(t_settings, dim_out)

    def call(self, split1, split2, condition, inverse=False, **kwargs):
        """Performs one pass through an affine coupling layer (either inverse or forward).

        Parameters
        ----------
        split1      : tf.Tensor of shape (batch_size, ..., input_dim//2)
            The first partition of the input vector(s)
        split2      : tf.Tensor of shape (batch_size, ..., ceil[input_dim//2])
            The second partition of the input vector(s)
        condition   : tf.Tensor or None
            The conditioning data of interest, for instance, x = summary_fun(x), shape (batch_size, ...).
            If ``condition is None``, then the layer reduces to an unconditional coupling.
        inverse     : bool, optional, default: False
            Flag indicating whether to run the block forward or backward.

        Returns
        -------
        (z, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            z shape: (batch_size, ..., input_dim//2), log_det_J shape: (batch_size, ...)

        target          :  tf.Tensor
            If inverse=True: The back-transformed z, shape (batch_size, ..., inp_dim//2)
        """

        if not inverse:
            return self._forward(split1, split2, condition, **kwargs)
        return self._inverse(split1, split2, condition, **kwargs)

    def _forward(self, u1, u2, condition, **kwargs):
        """Performs a forward pass through the coupling layer. Used internally by the instance.

        Parameters
        ----------
        v1        : tf.Tensor of shape (batch_size, ..., dim_1)
            The first partition of the input
        v2        : tf.Tensor of shape (batch_size, ..., dim_2)
            The second partition of the input
        condition : tf.Tensor of shape (batch_size, ..., dim_condition) or None
            The optional conditioning vector. Batch size must match the batch size
            of the partitions.

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            The transformed input and the corresponding Jacobian of the transformation.
        """

        s = self.scale(u2, condition, **kwargs)
        if self.soft_clamp is not None:
            s = (2.0 * self.soft_clamp / PI_CONST) * tf.math.atan(s / self.soft_clamp)
        t = self.translate(u2, condition, **kwargs)
        v = u1 * tf.math.exp(s) + t
        log_det_J = tf.reduce_sum(s, axis=-1)
        return v, log_det_J

    def _inverse(self, v1, v2, condition, **kwargs):
        """Performs an inverse pass through the affine coupling block. Used internally by the instance.

        Parameters
        ----------
        v1        : tf.Tensor of shape (batch_size, ..., dim_1)
            The first partition of the latent vector
        v2        : tf.Tensor of shape (batch_size, ..., dim_2)
            The second partition of the latent vector
        condition : tf.Tensor of shape (batch_size, ..., dim_condition)
            The optional conditioning vector. Batch size must match the batch size
            of the partitions.

        Returns
        -------
        u  :  tf.Tensor of shape (batch_size, ..., dim_1)
            The back-transformed input.
        """

        s = self.scale(v1, condition, **kwargs)
        if self.soft_clamp is not None:
            s = (2.0 * self.soft_clamp / PI_CONST) * tf.math.atan(s / self.soft_clamp)
        t = self.translate(v1, condition, **kwargs)
        u = (v2 - t) * tf.math.exp(-s)
        return u


class SplineCoupling(tf.keras.Model):
    """Implements a conditional spline coupling block according to [1, 2], with additional
    options, such as residual blocks or Monte Carlo Dropout.

    [1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
    Neural spline flows. Advances in Neural Information Processing Systems, 32.

    [2] Ardizzone, L., Lüth, C., Kruse, J., Rother, C., & Köthe, U. (2019).
    Guided image generation with conditional invertible neural networks.
    arXiv preprint arXiv:1907.02392.

    Implement only rational quadratic splines (RQS), since these appear to work
    best in practice and lead to stable training.
    """

    def __init__(self, dim_out, settings_dict, **kwargs):
        """Creates one half of a spline coupling layer to be used as part of a ``CouplingLayer`` in
        an ``InvertibleNetwork`` instance.

        Parameters
        ----------

        dim_out       : int
            The output dimensionality of the coupling layer.
        settings_dict : dict
            The settings for the inner networks. Defaults will use:
            ``settings_dict={
                "dense_args"     : dict(units=128, activation="relu"),
                "num_dense"      : 2,
                "spec_norm"      : False,
                "mc_dropout"     : False,
                "dropout"        : True,
                "residual"       : False,
                "dropout_prob"   : 0.05,
                "bins"           : 16,
                "default_domain" : (-5., 5., -5., 5.)
            }
            ``
        """
        super().__init__(**kwargs)

        self.dim_out = dim_out
        self.bins = settings_dict["bins"]
        self.default_domain = settings_dict["default_domain"]
        self.spline_params_counts = {
            "left_edge": 1,
            "bottom_edge": 1,
            "widths": self.bins,
            "heights": self.bins,
            "derivatives": self.bins - 1,
        }
        self.num_total_spline_params = sum(self.spline_params_counts.values()) * self.dim_out

        # Internal network (learnable spline parameters)
        self.net = DenseCouplingNet(settings_dict, self.num_total_spline_params)

    def call(self, split1, split2, condition, inverse=False, **kwargs):
        """Performs one pass through a spline coupling layer (either inverse or forward).

        Parameters
        ----------
        split1      : tf.Tensor of shape (batch_size, ..., input_dim//2)
            The first partition of the input vector(s)
        split2      : tf.Tensor of shape (batch_size, ..., ceil[input_dim//2])
            The second partition of the input vector(s)
        condition   : tf.Tensor or None
            The conditioning data of interest, for instance, x = summary_fun(x), shape (batch_size, ...).
            If ``condition is None``, then the layer recuces to an unconditional coupling.
        inverse     : bool, optional, default: False
            Flag indicating whether to run the block forward or backward.

        Returns
        -------
        (z, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            z shape: (batch_size, ..., input_dim//2), log_det_J shape: (batch_size, ...)

        target          :  tf.Tensor
            If inverse=True: The back-transformed z, shape (batch_size, ..., inp_dim//2)
        """

        if not inverse:
            return self._forward(split1, split2, condition, **kwargs)
        return self._inverse(split1, split2, condition, **kwargs)

    def _forward(self, u1, u2, condition, **kwargs):
        """Performs a forward pass through the spline coupling layer. Used internally by the instance.

        Parameters
        ----------
        v1        : tf.Tensor of shape (batch_size, ..., dim_1)
            The first partition of the input
        v2        : tf.Tensor of shape (batch_size, ..., dim_2)
            The second partition of the input
        condition : tf.Tensor of shape (batch_size, ..., dim_condition) or None
            The optional conditioning vector. Batch size must match the batch size
            of the partitions.

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            The transformed input and the corresponding Jacobian of the transformation.
        """

        spline_params = self.net(u2, condition, **kwargs)
        spline_params = self._semantic_spline_parameters(spline_params)
        spline_params = self._constrain_parameters(spline_params)
        v, log_det_J = self._calculate_spline(u1, spline_params, inverse=False)
        return v, log_det_J

    def _inverse(self, v1, v2, condition, **kwargs):
        """Performs an inverse pass through the coupling block. Used internally by the instance.

        Parameters
        ----------
        v1        : tf.Tensor of shape (batch_size, ..., dim_1)
            The first partition of the latent vector
        v2        : tf.Tensor of shape (batch_size, ..., dim_2)
            The second partition of the latent vector
        condition : tf.Tensor of shape (batch_size, ..., dim_condition)
            The optional conditioning vector. Batch size must match the batch size
            of the partitions.

        Returns
        -------
        u  :  tf.Tensor of shape (batch_size, ..., dim_1)
            The back-transformed input.
        """

        spline_params = self.net(v1, condition, **kwargs)
        spline_params = self._semantic_spline_parameters(spline_params)
        spline_params = self._constrain_parameters(spline_params)
        u = self._calculate_spline(v2, spline_params, inverse=True)
        return u

    def _calculate_spline(self, target, spline_params, inverse=False):
        """Computes both directions of a rational quadratic spline (RQS) as in:
        https://github.com/vislearn/FrEIA/blob/master/FrEIA/modules/splines/rational_quadratic.py

        At this point, ``spline_params`` represents a tuple with the parameters of the RQS learned
        by the internal neural network (given optional conditional information).

        Parameters
        ----------
        target         : tf.Tensor of shape (batch_size, ..., dim_2)
            The target partition of the input vector to transform.
        spline_params  : tuple(tf.Tensor,...)
            A tuple with tensors corresponding to the learnbale spline features:
            (left_edge, bottom_edge, widths, heights, derivatives)
        inverse        : bool, optional, default: False
            Flag indicating whether to run the block forward or backward.

        Returns
        -------
        (result, log_det_J) :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            result shape: (batch_size, ..., dim_2), log_det_J shape: (batch_size, ...)

        result              :  tf.Tensor
            If inverse=True: The back-transformed latent, shape (batch_size, ..., dim_2)
        """

        # Extract all learnable parameters
        left_edge, bottom_edge, widths, heights, derivatives = spline_params

        # Placeholders for results
        result = tf.zeros_like(target)
        log_jac = tf.zeros_like(target)

        total_width = tf.reduce_sum(widths, axis=-1, keepdims=True)
        total_height = tf.reduce_sum(heights, axis=-1, keepdims=True)

        knots_x = tf.concat([left_edge, left_edge + tf.math.cumsum(widths, axis=-1)], axis=-1)
        knots_y = tf.concat([bottom_edge, bottom_edge + tf.math.cumsum(heights, axis=-1)], axis=-1)

        # Determine which targets are in domain and which are not
        if not inverse:
            target_in_domain = tf.logical_and(knots_x[..., 0] < target, target <= knots_x[..., -1])
            higher_indices = tf.searchsorted(knots_x, target[..., None])
        else:
            target_in_domain = tf.logical_and(knots_y[..., 0] < target, target <= knots_y[..., -1])
            higher_indices = tf.searchsorted(knots_y, target[..., None])
        target_in = target[target_in_domain]
        target_in_idx = tf.where(target_in_domain)
        target_out = target[~target_in_domain]
        target_out_idx = tf.where(~target_in_domain)

        # In-domain computation
        if tf.size(target_in_idx) > 0:
            # Index crunching
            higher_indices = tf.gather_nd(higher_indices, target_in_idx)
            higher_indices = tf.cast(higher_indices, tf.int32)
            lower_indices = higher_indices - 1
            lower_idx_tuples = tf.concat([tf.cast(target_in_idx, tf.int32), lower_indices], axis=-1)
            higher_idx_tuples = tf.concat([tf.cast(target_in_idx, tf.int32), higher_indices], axis=-1)

            # Spline computation
            dk = tf.gather_nd(derivatives, lower_idx_tuples)
            dkp = tf.gather_nd(derivatives, higher_idx_tuples)
            xk = tf.gather_nd(knots_x, lower_idx_tuples)
            xkp = tf.gather_nd(knots_x, higher_idx_tuples)
            yk = tf.gather_nd(knots_y, lower_idx_tuples)
            ykp = tf.gather_nd(knots_y, higher_idx_tuples)
            x = target_in
            dx = xkp - xk
            dy = ykp - yk
            sk = dy / dx
            xi = (x - xk) / dx

            # Forward pass
            if not inverse:
                numerator = dy * (sk * xi**2 + dk * xi * (1 - xi))
                denominator = sk + (dkp + dk - 2 * sk) * xi * (1 - xi)
                result_in = yk + numerator / denominator
                # Log Jacobian for in-domain
                numerator = sk**2 * (dkp * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
                denominator = (sk + (dkp + dk - 2 * sk) * xi * (1 - xi)) ** 2
                log_jac_in = tf.math.log(numerator + 1e-10) - tf.math.log(denominator + 1e-10)
                log_jac = tf.tensor_scatter_nd_update(log_jac, target_in_idx, log_jac_in)
            # Inverse pass
            else:
                y = x
                a = dy * (sk - dk) + (y - yk) * (dkp + dk - 2 * sk)
                b = dy * dk - (y - yk) * (dkp + dk - 2 * sk)
                c = -sk * (y - yk)
                discriminant = tf.maximum(b**2 - 4 * a * c, 0.0)
                xi = 2 * c / (-b - tf.math.sqrt(discriminant))
                result_in = xi * dx + xk

            result = tf.tensor_scatter_nd_update(result, target_in_idx, result_in)

        # Out-of-domain
        if tf.size(target_out_idx) > 1:
            scale = total_height / total_width
            shift = bottom_edge - scale * left_edge
            scale_out = tf.gather_nd(scale, target_out_idx)
            shift_out = tf.gather_nd(shift, target_out_idx)

            if not inverse:
                result_out = scale_out * target_out[..., None] + shift_out
                # Log Jacobian for out-of-domain points
                log_jac_out = tf.math.log(scale_out + 1e-10)
                log_jac_out = tf.squeeze(log_jac_out, axis=-1)
                log_jac = tf.tensor_scatter_nd_update(log_jac, target_out_idx, log_jac_out)
            else:
                result_out = (target_out[..., None] - shift_out) / scale_out

            result_out = tf.squeeze(result_out, axis=-1)
            result = tf.tensor_scatter_nd_update(result, target_out_idx, result_out)

        if not inverse:
            return result, tf.reduce_sum(log_jac, axis=-1)
        return result

    def _semantic_spline_parameters(self, parameters):
        """Builds a tuple of tensors from the output of the coupling net.

        Parameters
        ----------
        parameters    : tf.Tensor of shape (batch_size, ..., num_spline_parameters)
            All learnable spline parameters packed in a single tensor, which will be
            partitioned according to the role of each spline parameter.

        Returns
        -------
        parameters    : tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor)
            The partitioned spline parameters according to their role in the spline computation.
        """

        shape = tf.shape(parameters)
        if len(shape) == 2:
            new_shape = (shape[0], self.dim_out, -1)
        elif len(shape) == 3:
            new_shape = (shape[0], shape[1], self.dim_out, -1)
        else:
            raise NotImplementedError("Spline flows can currently only operate on 2D and 3D inputs!")
        parameters = tf.reshape(parameters, new_shape)
        parameters = tf.split(parameters, list(self.spline_params_counts.values()), axis=-1)
        return parameters

    def _constrain_parameters(self, parameters):
        """Takes care of zero spline parameters due to zeros kernel initializer and
        applies parameter constraints for stability.

        Parameters
        ----------
        parameters : tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor)
            The unconstrained spline parameters.

        Returns
        -------
        parameters : tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor)
            The constrained spline parameters.
        """

        left_edge, bottom_edge, widths, heights, derivatives = parameters

        # Set lower corners of domain relative to default domain
        left_edge = left_edge + self.default_domain[0]
        bottom_edge = bottom_edge + self.default_domain[2]

        # Compute default widths and heights
        default_width = (self.default_domain[1] - self.default_domain[0]) / self.bins
        default_height = (self.default_domain[3] - self.default_domain[2]) / self.bins

        # Compute shifts for softplus function
        xshift = tf.math.log(tf.math.exp(default_width) - 1)
        yshift = tf.math.log(tf.math.exp(default_height) - 1)

        # Constrain widths and heights to be positive
        widths = tf.math.softplus(widths + xshift)
        heights = tf.math.softplus(heights + yshift)

        # Compute spline derivatives
        shift = tf.math.log(EULER_CONST - 1.0)
        derivatives = tf.nn.softplus(derivatives + shift)

        # Add in edge derivatives
        total_height = tf.reduce_sum(heights, axis=-1, keepdims=True)
        total_width = tf.reduce_sum(widths, axis=-1, keepdims=True)
        scale = total_height / total_width
        derivatives = tf.concat([scale, derivatives, scale], axis=-1)
        return left_edge, bottom_edge, widths, heights, derivatives


class CouplingLayer(tf.keras.Model):
    """General wrapper for a coupling layer (either affine or spline) with different settings."""

    def __init__(
        self,
        latent_dim,
        coupling_settings=None,
        coupling_design="affine",
        permutation="fixed",
        use_act_norm=True,
        act_norm_init=None,
        **kwargs,
    ):
        """Creates an invertible coupling layers instance with the provided hyperparameters.

        Parameters
        ----------
        latent_dim            : int
            The dimensionality of the latent space (equal to the dimensionality of the target variable)
        coupling_settings     : dict or None, optional, default: None
            The coupling network settings to pass to the internal coupling layers. See ``default_settings``
            for the required entries.
        coupling_design       : str or callable, optional, default: 'affine'
            The type of internal coupling network to use. Must be in ['affine', 'spline'].
            In general, spline couplings run slower than affine couplings, but require fewers coupling
            layers. Spline couplings may work best with complex (e.g., multimodal) low-dimensional
            problems. The difference will become less and less pronounced as we move to higher dimensions.
        permutation           : str or None, optional, default: 'fixed'
            Whether to use permutations between coupling layers. Highly recommended if ``num_coupling_layers > 1``
            Important: Must be in ['fixed', 'learnable', None]
        use_act_norm          : bool, optional, default: True
            Whether to use activation normalization after each coupling layer. Recommended to keep default.
        act_norm_init         : np.ndarray of shape (num_simulations, num_params) or None, optional, default: None
            Optional data-dependent initialization for the internal ``ActNorm`` layers.
        **kwargs              : dict
            Optional keyword arguments (e.g., name) passed to the tf.keras.Model __init__ method.
        """

        super().__init__(**kwargs)

        # Set dimensionality attributes
        self.latent_dim = latent_dim
        self.dim_out1 = self.latent_dim // 2
        self.dim_out2 = self.latent_dim // 2 if self.latent_dim % 2 == 0 else self.latent_dim // 2 + 1

        # Determine coupling net settings
        if coupling_settings is None:
            user_dict = dict()
        elif isinstance(coupling_settings, dict):
            user_dict = coupling_settings
        else:
            raise ConfigurationError("coupling_net_settings argument must be None or a dict!")

        # Determine type of coupling (affine or spline) and build settings
        if coupling_design == "affine":
            coupling_type = AffineCoupling
            coupling_settings = build_meta_dict(
                user_dict=user_dict, default_setting=default_settings.DEFAULT_SETTING_AFFINE_COUPLING
            )
        elif coupling_design == "spline":
            coupling_type = SplineCoupling
            coupling_settings = build_meta_dict(
                user_dict=user_dict, default_setting=default_settings.DEFAULT_SETTING_SPLINE_COUPLING
            )
        else:
            raise NotImplementedError('coupling_design must be in ["affine", "spline"]')

        # Two-in-one coupling block (i.e., no inactive part after a forward pass)
        self.net1 = coupling_type(self.dim_out1, coupling_settings)
        self.net2 = coupling_type(self.dim_out2, coupling_settings)

        # Optional (learnable or fixed) permutation
        if permutation not in ["fixed", "learnable", None]:
            raise ConfigurationError('Argument permutation should be in ["fixed", "learnable", None]')
        if permutation == "fixed":
            self.permutation = Permutation(self.latent_dim)
            self.permutation.trainable = False
        elif permutation == "learnable":
            self.permutation = Orthogonal(self.latent_dim)
        else:
            self.permutation = None

        # Optional learnable activation normalization
        if use_act_norm:
            self.act_norm = ActNorm(latent_dim, act_norm_init)
        else:
            self.act_norm = None

    def call(self, target_or_z, condition, inverse=False, **kwargs):
        """Performs one pass through a the affine coupling layer (either inverse or forward).

        Parameters
        ----------
        target_or_z      : tf.Tensor
            The estimation quantites of interest or latent representations z ~ p(z), shape (batch_size, ...)
        condition        : tf.Tensor or None
            The conditioning data of interest, for instance, x = summary_fun(x), shape (batch_size, ...).
            If `condition is None`, then the layer recuces to an unconditional ACL.
        inverse          : bool, optional, default: False
            Flag indicating whether to run the block forward or backward.

        Returns
        -------
        (z, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            z shape: (batch_size, inp_dim), log_det_J shape: (batch_size, )

        target          :  tf.Tensor
            If inverse=True: The back-transformed z, shape (batch_size, inp_dim)

        Important
        ---------
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``
        """

        if not inverse:
            return self.forward(target_or_z, condition, **kwargs)
        return self.inverse(target_or_z, condition, **kwargs)

    def forward(self, target, condition, **kwargs):
        """Performs a forward pass through a coupling layer with an optinal `Permutation` and `ActNorm` layers.

        Parameters
        ----------
        target     : tf.Tensor
            The estimation quantities of interest, for instance, parameter vector of shape (batch_size, theta_dim)
        condition  : tf.Tensor or None
            The conditioning vector of interest, for instance, x = summary(x), shape (batch_size, summary_dim)
            If `None`, transformation amounts to unconditional estimation.

        Returns
        -------
        (z, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            The transformed input and the corresponding Jacobian of the transformation.
        """

        # Initialize log_det_Js accumulator
        log_det_Js = tf.zeros(1)

        # Normalize activation, if specified
        if self.act_norm is not None:
            target, log_det_J_act = self.act_norm(target)
            log_det_Js += log_det_J_act

        # Permute, if indicated
        if self.permutation is not None:
            target = self.permutation(target)
        if self.permutation.trainable:
            target, log_det_J_p = target
            log_det_Js += log_det_J_p

        # Pass through coupling layer
        latent, log_det_J_c = self._forward(target, condition, **kwargs)
        log_det_Js += log_det_J_c
        return latent, log_det_Js

    def inverse(self, latent, condition, **kwargs):
        """Performs an inverse pass through a coupling layer with an optinal `Permutation` and `ActNorm` layers.

        Parameters
        ----------
        z          : tf.Tensor
            latent variables z ~ p(z), shape (batch_size, theta_dim)
        condition  : tf.Tensor or None
            The conditioning vector of interest, for instance, x = summary(x), shape (batch_size, summary_dim).
            If `None`, transformation amounts to unconditional estimation.

        Returns
        -------
        target  :  tf.Tensor
            The back-transformed latent variable z.
        """

        target = self._inverse(latent, condition, **kwargs)
        if self.permutation is not None:
            target = self.permutation(target, inverse=True)
        if self.act_norm is not None:
            target = self.act_norm(target, inverse=True)
        return target

    def _forward(self, target, condition, **kwargs):
        """Performs a forward pass through the coupling layer. Used internally by the instance.

        Parameters
        ----------
        target     : tf.Tensor
            The estimation quantities of interest, for instance, parameter vector of shape (batch_size, theta_dim)
        condition  : tf.Tensor or None
            The conditioning vector of interest, for instance, x = summary(x), shape (batch_size, summary_dim)
            If `None`, transformation amounts to unconditional estimation.

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            The transformed input and the corresponding Jacobian of the transformation.
        """

        # Split input along last axis and perform forward coupling
        u1, u2 = tf.split(target, [self.dim_out1, self.dim_out2], axis=-1)
        v1, log_det_J1 = self.net1(u1, u2, condition, inverse=False, **kwargs)
        v2, log_det_J2 = self.net2(u2, v1, condition, inverse=False, **kwargs)
        v = tf.concat((v1, v2), axis=-1)

        # Compute log determinat of the Jacobians from both splits
        log_det_J = log_det_J1 + log_det_J2
        return v, log_det_J

    def _inverse(self, latent, condition, **kwargs):
        """Performs an inverse pass through the coupling block. Used internally by the instance.

        Parameters
        ----------
        latent       : tf.Tensor
            latent variables z ~ p(z), shape (batch_size, theta_dim)
        condition    : tf.Tensor or None
            The conditioning vector of interest, for instance, x = summary(x), shape (batch_size, summary_dim).
            If `None`, transformation amounts to unconditional estimation.

        Returns
        -------
        u  :  tf.Tensor
            The back-transformed input.
        """

        # Split input along last axis and perform inverse coupling
        v1, v2 = tf.split(latent, [self.dim_out1, self.dim_out2], axis=-1)
        u2 = self.net2(v1, v2, condition, inverse=True, **kwargs)
        u1 = self.net1(u2, v1, condition, inverse=True, **kwargs)
        u = tf.concat((u1, u2), axis=-1)
        return u
