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

import tensorflow as tf
import tensorflow_probability as tfp

import bayesflow.default_settings as defaults
from bayesflow.computational_utilities import compute_jacobian_trace
from bayesflow.exceptions import SummaryStatsError
from bayesflow.helper_networks import MCDropout
from bayesflow.losses import mmd_summary_space


class DriftNetwork(tf.keras.Model):
    """Implements a learnable velocity field for a neural ODE. Will typically be used
    in conjunction with a ``RectifyingFlow`` instance, as proposed by [1] in the context
    of unconditional image generation.

    [1] Liu, X., Gong, C., & Liu, Q. (2022).
    Flow straight and fast: Learning to generate and transfer data with rectified flow.
    arXiv preprint arXiv:2209.03003.
    """

    def __init__(
        self, target_dim, num_dense=3, dense_args=None, dropout=True, mc_dropout=False, dropout_prob=0.05, **kwargs
    ):
        """Creates a learnable velocity field instance to be used in the context of rectifying
        flows or neural ODEs.

        [1] Liu, X., Gong, C., & Liu, Q. (2022).
        Flow straight and fast: Learning to generate and transfer data with rectified flow.
        arXiv preprint arXiv:2209.03003.

        Parameters
        ----------
        target_dim   : int
            The problem dimensionality (e.g., in parameter estimation, the number of parameters)
        num_dense    : int, optional, default: 3
            The number of hidden layers for the inner fully-connected network
        dense_args   : dict or None, optional, default: None
            The arguments to be passed to ``tf.keras.layers.Dense`` constructor. If None, default settings
            will be fetched from ``bayesflow.default_settings``.
        dropout      : bool, optional, default: True
            Whether to use dropout in-between the hidden layers.
        mc_dropout   : bool, optional, default: False
            Whether to use dropout Monte Carlo dropout (i.e., Bayesian approximation) during inference
        dropout_prob : float in (0, 1), optional, default: 0.05
            The dropout probability. Only has effecft if ``dropout=True`` or ``mc_dropout=True``
        **kwargs     : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model.__init__`` method.
        """

        super().__init__(**kwargs)

        self.latent_dim = target_dim
        if dense_args is None:
            dense_args = defaults.DEFAULT_SETTING_DENSE_RECT
        self.net = tf.keras.Sequential()
        for _ in range(num_dense):
            self.net.add(tf.keras.layers.Dense(**dense_args))
            if mc_dropout:
                self.net.add(MCDropout(dropout_prob))
            elif dropout:
                self.net.add(tf.keras.layers.Dropout(dropout_prob))
            else:
                pass
        self.net.add(tf.keras.layers.Dense(self.latent_dim))
        self.net.build(input_shape=())

    def call(self, target_vars, latent_vars, time, condition, **kwargs):
        """Performs a linear interpolation between target and latent variables
        over time (i.e., a single ODE step during training).

        Parameters
        ----------
        target_vars : tf.Tensor of shape (batch_size, ..., num_targets)
            The variables of interest (e.g., parameters) over which we perform inference.
        latent_vars : tf.Tensor of shape (batch_size, ..., num_targets)
            The sampled random variates from the base distribution.
        time        : tf.Tensor of shape (batch_size, ..., 1)
            A vector of time indices in (0, 1)
        condition   : tf.Tensor of shape (batch_size, ..., condition_dim)
            The optional conditioning variables (e.g., as returned by a summary network)
        **kwargs    : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model`` call() method
        """

        diff = target_vars - latent_vars
        wdiff = time * target_vars + (1 - time) * latent_vars
        drift = self.drift(wdiff, time, condition, **kwargs)
        return diff, drift

    def drift(self, target_t, time, condition, **kwargs):
        """Returns the drift at target_t time given optional condition(s).

        Parameters
        ----------
        target_t    : tf.Tensor of shape (batch_size, ..., num_targets)
            The variables of interest (e.g., parameters) over which we perform inference.
        time        : tf.Tensor of shape (batch_size, ..., 1)
            A vector of time indices in (0, 1)
        condition   : tf.Tensor of shape (batch_size, ..., condition_dim)
            The optional conditioning variables (e.g., as returned by a summary network)
        **kwargs    : dict, optional, default: {}
            Optional keyword arguments passed to the drift network.
        """

        if condition is not None:
            inp = tf.concat([target_t, condition, time], axis=-1)
        else:
            inp = tf.concat([target_t, time], axis=-1)
        return self.net(inp, **kwargs)


class RectifiedDistribution(tf.keras.Model):
    """Implements a rectifying flows according to [1]. To be used as an alternative
    to a normalizing flow in a BayesFlow pipeline.

    [1] Liu, X., Gong, C., & Liu, Q. (2022).
    Flow straight and fast: Learning to generate and transfer data with rectified flow.
    arXiv preprint arXiv:2209.03003.
    """

    def __init__(self, drift_net, summary_net=None, latent_dist=None, loss_fun=None, summary_loss_fun=None, **kwargs):
        """Initializes a composite neural network to represent an amortized approximate posterior through
        for a rectifying flow.

        Parameters
        ----------
        drift_net         : tf.keras.Model
            A neural network for the velocity field (drift) of the learnable ODE
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network to compress non-vector data structures.
        latent_dist       : callable or None, optional, default: None
            The latent distribution towards which to optimize the networks. Defaults to
            a multivariate unit Gaussian.
        loss_fun          : callable or None, optional, default: None
            The loss function for "rectifying" the velocity field. If ``None``, defaults
            to tf.keras.losses.logcosh. Sensible alternatives are MSE (as in [])
        summary_loss_fun  : callable, str, or None, optional, default: None
            The loss function which accepts the outputs of the summary network. If ``None``, no loss is provided
            and the summary space will not be shaped according to a known distribution (see [2]).
            If ``summary_loss_fun='MMD'``, the default loss from [2] will be used.
        **kwargs          : dict, optional, default: {}
            Additional keyword arguments passed to the ``__init__`` method of a ``tf.keras.Model`` instance.

        Important
        ----------
        - If no ``summary_net`` is provided, then the output dictionary of your generative model should not contain
        any ``summary_conditions``, i.e., ``summary_conditions`` should be set to ``None``, otherwise these will be ignored.
        """

        super().__init__(**kwargs)

        self.drift_net = drift_net
        self.summary_net = summary_net
        self.latent_dim = drift_net.latent_dim
        self.latent_dist = self._determine_latent_dist(latent_dist)
        self.loss_fun = self._determine_loss(loss_fun)
        self.summary_loss = self._determine_summary_loss(summary_loss_fun)

    def call(self, input_dict, return_summary=False, num_eval_points=1, **kwargs):
        """Performs a forward pass through the summary and drift network given an input dictionary.

        Parameters
        ----------
        input_dict      : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``targets``            - the latent model parameters over which a condition density is learned
            ``summary_conditions`` - the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        return_summary  : bool, optional, default: False
            A flag which determines whether the learnable data summaries (representations) are returned or not.
        num_eval_points : int, optional, default: 1
            The number of time points for evaluating the noisy estimator. Values larger than the default 1
            may reduce the variance of the estimator, but may lead to increased memory demands, since an
            additional dimension is added at axis 1 of all tensors.
        **kwargs        : dict, optional, default: {}
            Additional keyword arguments passed to the networks
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        net_out or (net_out, summary_out)
        """

        # Concatenate conditions, if given
        summary_out, full_cond = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            **kwargs,
        )

        # Extract target variables
        target_vars = input_dict[defaults.DEFAULT_KEYS["parameters"]]

        # Extract batch size (autograph friendly)
        batch_size = tf.shape(target_vars)[0]

        # Sample latent variables
        latent_vars = self.latent_dist.sample(batch_size)

        # Do a little trick for less noisy estimator, if evals > 1
        if num_eval_points > 1:
            target_vars = tf.stack([target_vars] * num_eval_points, axis=1)
            latent_vars = tf.stack([latent_vars] * num_eval_points, axis=1)
            full_cond = tf.stack([full_cond] * num_eval_points, axis=1)
            # Sample time
            time = tf.random.uniform((batch_size, num_eval_points, 1))
        else:
            time = tf.random.uniform((batch_size, 1))

        # Compute drift
        net_out = self.drift_net(target_vars, latent_vars, time, full_cond, **kwargs)

        # Return summary outputs or not, depending on parameter
        if return_summary:
            return net_out, summary_out
        return net_out

    def compute_loss(self, input_dict, **kwargs):
        """Computes the loss of the posterior amortizer given an input dictionary, which will
        typically be the output of a Bayesian ``GenerativeModel`` instance.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``targets``            - the latent variables over which a condition density is learned
            ``summary_conditions`` - the conditioning variables that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        **kwargs   : dict, optional, default: {}
            Additional keyword arguments passed to the networks
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        total_loss : tf.Tensor of shape (1,) - the total computed loss given input variables
        """

        net_out, sum_out = self(input_dict, return_summary=True, **kwargs)
        diff, drift = net_out
        loss = self.loss_fun(diff, drift)

        # Case summary loss should be computed
        if self.summary_loss is not None:
            sum_loss = self.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Compute and return total loss
        total_loss = tf.reduce_mean(loss) + sum_loss
        return total_loss

    def sample(self, input_dict, n_samples, to_numpy=True, step_size=1e-3, **kwargs):
        """Generates random draws from the approximate posterior given a dictionary with conditonal variables.

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``
        step_size  : float, optional, default: 0.01
            The step size for the stochastic Euler solver.
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_data_sets, n_samples, n_params)
            The sampled parameters from the approximate posterior of each data set
        """

        # Compute condition (direct, summary, or both)
        _, conditions = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs,
        )
        n_data_sets = tf.shape(conditions)[0]

        # Sample initial latent variables -> shape (n_data_sets, n_samples, latent_dim)
        latent_vars = self.latent_dist.sample((n_data_sets, n_samples))

        # Replicate conditions and solve ODEs simulatenously
        conditions = tf.stack([conditions] * n_samples, axis=1)
        post_samples = self._solve_euler(latent_vars, conditions, step_size, **kwargs)

        # Remove trailing first dimension in the single data case
        if n_data_sets == 1:
            post_samples = tf.squeeze(post_samples, axis=0)

        # Return numpy version of tensor or tensor itself
        if to_numpy:
            return post_samples.numpy()
        return post_samples

    def log_density(self, input_dict, to_numpy=True, step_size=1e-3, **kwargs):
        """Computes the log density..."""

        # Compute condition (direct, summary, or both)
        _, conditions = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs,
        )

        # Extract targets
        target_vars = input_dict[defaults.DEFAULT_KEYS["parameters"]]

        # Reverse ODE and log pdf computation with the trace method
        latents, trace = self._solve_euler_inv(target_vars, conditions, step_size, **kwargs)
        lpdf = self.latent_dist.log_prob(latents) + trace

        # Return numpy version of tensor or tensor itself
        if to_numpy:
            return lpdf.numpy()
        return lpdf

    def _solve_euler(self, latent_vars, condition, dt=1e-3, **kwargs):
        """Simple stochastic parallel Euler solver."""

        num_steps = int(1 / dt)
        time_vec = tf.zeros((tf.shape(latent_vars)[0], tf.shape(latent_vars)[1], 1))
        target = tf.identity(latent_vars)
        for _ in range(num_steps + 1):
            target += self.drift_net.drift(target, time_vec, condition, **kwargs) * dt
            time_vec += dt
        return target

    def _solve_euler_inv(self, targets, condition, dt=1e-3, **kwargs):
        """Solves the reverse ODE (negative direction of drift) and returns the trace."""

        def velocity(latents, drift, time_vec, condition, **kwargs):
            v = drift(latents, time_vec, condition, **kwargs)
            return v

        batch_size = tf.shape(targets)[0]
        num_samples = tf.shape(targets)[1]
        num_steps = int(1 / dt)
        time_vec = tf.ones((batch_size, num_samples, 1))
        trace = tf.zeros((batch_size, num_samples))
        latents = tf.identity(targets)
        for _ in range(num_steps + 1):
            f = partial(velocity, drift=self.drift_net.drift, time_vec=time_vec, condition=condition)
            drift_t, trace_t = compute_jacobian_trace(f, latents, **kwargs)
            latents -= drift_t * dt
            trace -= trace_t * dt
            time_vec -= dt
        return latents, trace

    def _compute_summary_condition(self, summary_conditions, direct_conditions, **kwargs):
        """Determines how to concatenate the provided conditions."""

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sum_condition = self.summary_net(summary_conditions, **kwargs)
        else:
            sum_condition = None

        # Concatenate learnable summaries with fixed summaries
        if sum_condition is not None and direct_conditions is not None:
            full_cond = tf.concat([sum_condition, direct_conditions], axis=-1)
        elif sum_condition is not None:
            full_cond = sum_condition
        elif direct_conditions is not None:
            full_cond = direct_conditions
        else:
            raise SummaryStatsError("Could not concatenarte or determine conditioning inputs...")
        return sum_condition, full_cond

    def _determine_latent_dist(self, latent_dist):
        """Determines which latent distribution to use and defaults to unit normal if ``None`` provided."""

        if latent_dist is None:
            return tfp.distributions.MultivariateNormalDiag(loc=[0.0] * self.latent_dim)
        else:
            return latent_dist

    def _determine_summary_loss(self, loss_fun):
        """Determines which summary loss to use if default `None` argument provided, otherwise return identity."""

        # If callable, return provided loss
        if loss_fun is None or callable(loss_fun):
            return loss_fun

        # If string, check for MMD or mmd
        elif type(loss_fun) is str:
            if loss_fun.lower() == "mmd":
                return mmd_summary_space
            else:
                raise NotImplementedError("For now, only 'mmd' is supported as a string argument for summary_loss_fun!")
        # Throw if loss type unexpected
        else:
            raise NotImplementedError(
                "Could not infer summary_loss_fun, argument should be of type (None, callable, or str)!"
            )

    def _determine_loss(self, loss_fun):
        """Determines which summary loss to use if default ``None`` argument provided, otherwise return identity."""

        if loss_fun is None:
            return tf.keras.losses.log_cosh
        return loss_fun
