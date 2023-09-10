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

import logging
from abc import ABC, abstractmethod
from functools import partial
from warnings import warn

logging.basicConfig()

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesflow.default_settings import DEFAULT_KEYS
from bayesflow.exceptions import ConfigurationError, SummaryStatsError
from bayesflow.helper_functions import check_tensor_sanity
from bayesflow.losses import log_loss, mmd_summary_space
from bayesflow.networks import EvidentialNetwork


class AmortizedTarget(ABC):
    """An abstract interface for an amortized learned distribution. Children should
    implement the following public methods:

    1. ``compute_loss(self, input_dict, **kwargs)``
    2. ``sample(input_dict, **kwargs)``
    3. ``log_prob(input_dict, **kwargs)``
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, input_dict, **kwargs):
        pass

    @abstractmethod
    def sample(self, input_dict, **kwargs):
        pass

    @abstractmethod
    def log_prob(self, input_dict, **kwargs):
        pass

    def _check_output_sanity(self, tensor):
        logger = logging.getLogger()
        check_tensor_sanity(tensor, logger)


class AmortizedPosterior(tf.keras.Model, AmortizedTarget):
    """A wrapper to connect an inference network for parameter estimation with an optional summary network
    as in the original BayesFlow set-up described in the paper:

    [1] Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020).
    BayesFlow: Learning complex stochastic models with invertible neural networks.
    IEEE Transactions on Neural Networks and Learning Systems.

    But also allowing for augmented functionality, such as model misspecification detection in summary space:

    [2] Schmitt, M., Bürkner, P. C., Köthe, U., & Radev, S. T. (2022).
    Detecting Model Misspecification in Amortized Bayesian Inference with Neural Networks
    arXiv preprint arXiv:2112.08866.

    And learning of fat-tailed posteriors with a Student-t latent pushforward density:

    [3] Jaini, P., Kobyzev, I., Yu, Y., & Brubaker, M. (2020, November).
    Tails of lipschitz triangular flows.
    In International Conference on Machine Learning (pp. 4673-4681). PMLR.

    [4] Alexanderson, S., & Henter, G. E. (2020).
    Robust model training and generalisation with Studentising flows.
    arXiv preprint arXiv:2006.06599.

    Serves as in interface for learning ``p(parameters | data, context).``
    """

    def __init__(
        self,
        inference_net,
        summary_net=None,
        latent_dist=None,
        latent_is_dynamic=False,
        summary_loss_fun=None,
        **kwargs,
    ):
        """Initializes a composite neural network to represent an amortized approximate posterior
        for a Bayesian generative model.

        Parameters
        ----------
        inference_net     : tf.keras.Model
            An (invertible) inference network which processes the outputs of a generative model
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network to compress non-vector data structures.
        latent_dist       : callable or None, optional, default: None
            The latent distribution towards which to optimize the networks. Defaults to
            a multivariate unit Gaussian.
        latent_is_dynamic : bool, optional, default: False
            If set to `True`, assumes that ``latent_dist`` is a function of the condtion and takes
            a different shape depending on the condition. Useful for more expressive transforms
            of complex distributions, such as fat-tailed or highly-multimodal distributions.

            Important: In the case of dynamic latents, the user is responsible that the
            latent is appropriately parameterized! If not using ``tensorflow_probability``,
            the ``latent_dist`` object needs to implement the following methods:
            - ``latent_dist(x).log_prob(z)`` and
            - ``latent_dist(x).sample(n_samples)``
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

        tf.keras.Model.__init__(self, **kwargs)

        self.inference_net = inference_net
        self.summary_net = summary_net
        self.latent_dim = self.inference_net.latent_dim
        self.latent_is_dynamic = latent_is_dynamic
        self.summary_loss = self._determine_summary_loss(summary_loss_fun)
        self.latent_dist = self._determine_latent_dist(latent_dist)

    def call(self, input_dict, return_summary=False, **kwargs):
        """Performs a forward pass through the summary and inference network given an input dictionary.

        Parameters
        ----------
        input_dict     : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``parameters``         - the latent model parameters over which a condition density is learned
            ``summary_conditions`` - the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        return_summary : bool, optional, default: False
            A flag which determines whether the learnable data summaries (representations) are returned or not.
        **kwargs       : dict, optional, default: {}
            Additional keyword arguments passed to the networks
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        net_out or (net_out, summary_out) : tuple of tf.Tensor
            the outputs of ``inference_net(theta, summary_net(x, c_s), c_d)``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J) or (sum_outputs, (z, log_det_J))`` if
            ``return_summary`` is set to True and a summary network is defined.``
        """

        # Concatenate conditions, if given
        summary_out, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(DEFAULT_KEYS["direct_conditions"]),
            **kwargs,
        )

        # Compute output of inference net
        net_out = self.inference_net(input_dict[DEFAULT_KEYS["parameters"]], full_cond, **kwargs)

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
            ``parameters``         - the latent model parameters over which a condition density is learned
            ``summary_conditions`` - the conditioning variables that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        **kwargs   : dict, optional, default: {}
            Additional keyword arguments passed to the networks
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        total_loss : tf.Tensor of shape (1,) - the total computed loss given input variables
        """

        # Get amortizer outputs
        net_out, sum_out = self(input_dict, return_summary=True, **kwargs)
        z, log_det_J = net_out

        # Case summary loss should be computed
        if self.summary_loss is not None:
            sum_loss = self.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Case dynamic latent space - function of summary conditions
        if self.latent_is_dynamic:
            logpdf = self.latent_dist(sum_out).log_prob(z)
        # Case _static latent space
        else:
            logpdf = self.latent_dist.log_prob(z)

        # Compute and return total loss
        total_loss = tf.reduce_mean(-logpdf - log_det_J) + sum_loss
        return total_loss

    def call_loop(self, input_list, return_summary=False, **kwargs):
        """Performs a forward pass through the summary and inference network given a list of dicts
        with the appropriate entries (i.e., as used for the standard call method).

        This method is useful when GPU memory is limited or data sets have a different (non-Tensor) structure.

        Parameters
        ----------
        input_list     : list of dicts, where each dict contains the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``parameters``         - the latent model parameters over which a condition density is learned
            ``summary_conditions`` - the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        return_summary : bool, optional, default: False
            A flag which determines whether the learnable data summaries (representations) are returned or not.
        **kwargs       : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        net_out or (net_out, summary_out) : tuple of tf.Tensor
            the outputs of ``inference_net(theta, summary_net(x, c_s), c_d)``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J) or (sum_outputs, (z, log_det_J)) if
            return_summary is set to True and a summary network is defined.``
        """

        outputs = []
        for forward_dict in input_list:
            outputs.append(self(forward_dict, return_summary, **kwargs))
        net_out = [tf.concat([o[i] for o in outputs], axis=0) for i in range(len(outputs[0]))]
        return tuple(net_out)

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """Generates random draws from the approximate posterior given a dictionary with conditonal variables.

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing at least one of the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``.
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_data_sets, n_samples, n_params)
            The sampled parameters from the approximate posterior of each data set
        """

        # Compute learnable summaries, if appropriate
        _, conditions = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs,
        )

        # Obtain number of data sets
        n_data_sets = conditions.shape[0]

        # Obtain random draws from the approximate posterior given conditioning variables
        # Case dynamic, assume tensorflow_probability instance, so need to reshape output from
        # (n_samples, n_data_sets, latent_dim) to (n_data_sets, n_samples, latent_dim)
        if self.latent_is_dynamic:
            z_samples = self.latent_dist(conditions).sample(n_samples)
            z_samples = tf.transpose(z_samples, (1, 0, 2))
        # Case _static latent - marginal samples from the specified dist
        else:
            z_samples = self.latent_dist.sample((n_data_sets, n_samples))

        # Obtain random draws from the approximate posterior given conditioning variables
        post_samples = self.inference_net.inverse(z_samples, conditions, training=False, **kwargs)

        # Only return 2D array, if first dimensions is 1
        if post_samples.shape[0] == 1:
            post_samples = post_samples[0]
        self._check_output_sanity(post_samples)

        # Return numpy version of tensor or tensor itself
        if to_numpy:
            return post_samples.numpy()
        return post_samples

    def sample_loop(self, input_list, n_samples, to_numpy=True, **kwargs):
        """Generates random draws from the approximate posterior given a list of dicts with conditonal variables.
        Useful when GPU memory is limited or data sets have a different (non-Tensor) structure.

        Parameters
        ----------
        input_list  : list of dictionaries, each dictionary having the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, n_params)
            The sampled parameters from the approximate posterior of each data set
        """

        post_samples = []
        for input_dict in input_list:
            post_samples.append(self.sample(input_dict, n_samples, to_numpy, **kwargs))
        if to_numpy:
            return np.concatenate(post_samples, axis=0)
        return tf.concat(post_samples, axis=0)

    def log_posterior(self, input_dict, to_numpy=True, **kwargs):
        """Calculates the approximate log-posterior of targets given conditional variables via
        the change-of-variable formula for a conditional normalizing flow.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``parameters``         : the latent model parameters over which a conditional density (i.e., a posterior) is learned
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that are directly passed to the inference network
        to_numpy   : bool, optional, default: True
            Flag indicating whether to return the lpdf values as a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs   : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        log_post   : tf.Tensor or np.ndarray of shape (batch_size, n_obs)
            the approximate log-posterior density of each each parameter
        """

        # Compute learnable summaries, if appropriate
        _, conditions = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs,
        )

        # Forward pass through the network
        z, log_det_J = self.inference_net.forward(
            input_dict[DEFAULT_KEYS["parameters"]], conditions, training=False, **kwargs
        )

        # Compute approximate log posterior
        # Case dynamic latent - function of conditions
        if self.latent_is_dynamic:
            log_post = self.latent_dist(conditions).log_prob(z) + log_det_J
        # Case _static latent - marginal samples from z
        else:
            log_post = self.latent_dist.log_prob(z) + log_det_J
        self._check_output_sanity(log_post)

        if to_numpy:
            return log_post.numpy()
        return log_post

    def log_prob(self, input_dict, to_numpy=True, **kwargs):
        """Identical to `log_posterior(input_dict, to_numpy, **kwargs)`."""

        return self.log_posterior(input_dict, to_numpy=to_numpy, **kwargs)

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
        """Determines which latent distribution to use and defaults to unit normal if None provided."""

        if latent_dist is None:
            return tfp.distributions.MultivariateNormalDiag(loc=[0.0] * self.latent_dim)
        else:
            return latent_dist

    def _determine_summary_loss(self, loss_fun):
        """Determines which summary loss to use if default `None` argument provided, otherwise return identity."""

        # Throw, if summary loss without a summary network provided
        if loss_fun is not None and self.summary_net is None:
            raise ConfigurationError('You need to provide a summary_net if you want to use a summary_loss_fun.')

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


class AmortizedLikelihood(tf.keras.Model, AmortizedTarget):
    """An interface for a surrogate model of a simulator, or an implicit likelihood
    ``p(data | parameters, context)``.
    """

    def __init__(self, surrogate_net, latent_dist=None, **kwargs):
        """Initializes a composite neural architecture representing an amortized emulator
        for the simulator (i.e., the implicit likelihood model).

        Parameters
        ----------
        surrogate_net : tf.keras.Model
            An (invertible) inference network which processes the outputs of the generative model.
        latent_dist       : callable or None, optional, default: None
            The latent distribution towards which to optimize the surrogate network outputs. Defaults to
            a multivariate unit Gaussian.
        **kwargs          : dict, optional, default: {}
            Additional keyword arguments passed to the ``__init__`` method of a ``tf.keras.Model`` instance.
        """

        tf.keras.Model.__init__(self, **kwargs)

        self.surrogate_net = surrogate_net
        self.latent_dim = self.surrogate_net.latent_dim
        self.latent_dist = self._determine_latent_dist(latent_dist)

    def call(self, input_dict, **kwargs):
        """Performs a forward pass through the summary and inference network.

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``observables`` - the observables over which a condition density is learned (i.e., the data)
            ``conditions``  - the conditioning variables that the directly passed to the inference network
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the network
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        net_out
            the outputs of ``surrogate_net(theta, summary_net(x, c_s), c_d)``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J)``.
        """

        net_out = self.surrogate_net(
            input_dict[DEFAULT_KEYS["observables"]], input_dict[DEFAULT_KEYS["conditions"]], **kwargs
        )
        return net_out

    def call_loop(self, input_list, **kwargs):
        """Performs a forward pass through the surrogate network given a list of dicts
        with the appropriate entries (i.e., as used for the standard call method).

        This method is useful when GPU memory is limited or data sets have a different (non-Tensor) structure.

        Parameters
        ----------
        input_list  : list of dicts, where each dict contains the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``observables`` - the observables over which a condition density is learned (i.e., the data)
            ``conditions``  - the conditioning variables that the directly passed to the inference network
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the network

        Returns
        -------
        net_out or (net_out, summary_out) : tuple of tf.Tensor
            the outputs of ``inference_net(theta, summary_net(x, c_s), c_d)``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J)``.
        """

        outputs = []
        for forward_dict in input_list:
            outputs.append(self(forward_dict, **kwargs))
        net_out = [tf.concat([o[i] for o in outputs], axis=0) for i in range(len(outputs[0]))]
        return tuple(net_out)

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """Generates `n_samples` random draws from the surrogate likelihood given input conditions.

        Parameters
        ----------

        input_dict  : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``conditions`` - the conditioning variables that are directly passed to the surrogate network
        n_samples   : int
            The number of posterior samples to obtain from the approximate posterior
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the network

        Returns
        -------
        lik_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, None)
            A simulated batch of observables from the surrogate likelihood.
        """

        # Extract condition
        conditions = input_dict[DEFAULT_KEYS["conditions"]]

        # Obtain number of data sets
        n_data_sets = conditions.shape[0]

        # Obtain random draws from the surrogate likelihood given conditioning variables
        z_samples = self.latent_dist.sample((n_data_sets, n_samples))

        # Obtain random draws from the surrogate likelihood given conditioning variables
        lik_samples = self.surrogate_net.inverse(z_samples, conditions, training=False, **kwargs)

        # Only return 2D array, if first dimensions is 1
        if lik_samples.shape[0] == 1:
            lik_samples = lik_samples[0]
        self._check_output_sanity(lik_samples)

        if to_numpy:
            return lik_samples.numpy()
        return lik_samples

    def sample_loop(self, input_list, n_samples, to_numpy=True, **kwargs):
        """Generates random draws from the surrogate network given a list of dicts with conditional variables.
        Useful when GPU memory is limited or data sets have a different (non-Tensor) structure.

        Parameters
        ----------
        input_list   : list of dictionaries, each dictionary having the following mandatory keys (default):
            ``conditions`` - the conditioning variables that the directly passed to the surrogate network
        n_samples    : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        to_numpy     : bool, optional, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`
        **kwargs     : dict, optional, default: {}
            Additional keyword arguments passed to the network

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_data_sets, n_samples, data_dim)
            the sampled parameters per data set
        """

        post_samples = []
        for input_dict in input_list:
            post_samples.append(self.sample(input_dict, n_samples, to_numpy, **kwargs))
        if to_numpy:
            return np.concatenate(post_samples, axis=0)
        return tf.concat(post_samples, axis=0)

    def log_likelihood(self, input_dict, to_numpy=True, **kwargs):
        """Calculates the approximate log-likelihood of targets given conditional variables via
        the change-of-variable formula for a conditional normalizing flow.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``observables`` - the variables over which a condition density is learned (i.e., the observables)
            ``conditions``  - the conditioning variables that the directly passed to the inference network
        to_numpy   : bool, optional, default: True
            Boolean flag indicating whether to return the log-lik values as a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs   : dict, optional, default: {}
            Additional keyword arguments passed to the network

        Returns
        -------
        log_lik    : tf.Tensor or np.ndarray of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        # Forward pass through the network
        z, log_det_J = self.surrogate_net.forward(
            input_dict[DEFAULT_KEYS["observables"]], input_dict[DEFAULT_KEYS["conditions"]], training=False, **kwargs
        )

        # Compute approximate log likelihood
        log_lik = self.latent_dist.log_prob(z) + log_det_J
        self._check_output_sanity(log_lik)

        # Convert tensor to numpy array, if specified
        if to_numpy:
            return log_lik.numpy()
        return log_lik

    def log_prob(self, input_dict, to_numpy=True, **kwargs):
        """Identical to `log_likelihood(input_dict, to_numpy, **kwargs)`."""

        return self.log_likelihood(input_dict, to_numpy=to_numpy, **kwargs)

    def compute_loss(self, input_dict, **kwargs):
        """Computes the loss of the amortized given input data provided in input_dict.

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing the following mandatory keys:
            ``data``        - the observables over which a condition density is learned (i.e., the observables)
            ``conditions``  - the conditioning variables that the directly passed to the surrogate network
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the network
            For instance, ``kwargs={'training': True}`` is passed automatically during simulation-based training.

        Returns
        -------
        loss        : tf.Tensor of shape (1,) - the total computed loss given input variables
        """

        z, log_det_J = self(input_dict, **kwargs)
        loss = tf.reduce_mean(-self.latent_dist.log_prob(z) - log_det_J)
        return loss

    def _determine_latent_dist(self, latent_dist):
        """Determines which latent distribution to use and defaults to unit normal if ``None`` provided."""

        if latent_dist is None:
            return tfp.distributions.MultivariateNormalDiag(loc=[0.0] * self.latent_dim)
        else:
            return latent_dist


class AmortizedPosteriorLikelihood(tf.keras.Model, AmortizedTarget):
    """An interface for jointly learning a surrogate model of the simulator and an approximate
    posterior given a generative model, as proposed by:

    [1] Radev, S. T., Schmitt, M., Pratz, V., Picchini, U., Köthe, U., & Bürkner, P. C. (2023).
    JANA: Jointly Amortized Neural Approximation of Complex Bayesian Models.
    arXiv preprint arXiv:2302.09125.
    """

    def __init__(self, amortized_posterior, amortized_likelihood, **kwargs):
        """Initializes a joint learner comprising an amortized posterior and an amortized emulator.

        Parameters
        ----------
        amortized_posterior  : an instance of AmortizedPosterior or a custom tf.keras.Model
            The generative neural posterior approximator
        amortized_likelihood : an instance of AmortizedLikelihood or a custom tf.keras.Model
            The generative neural likelihood approximator
        **kwargs          : dict, optional, default: {}
            Additional keyword arguments passed to the ``__init__`` method of a ``tf.keras.Model`` instance
        """

        tf.keras.Model.__init__(self, **kwargs)

        self.amortized_posterior = amortized_posterior
        self.amortized_likelihood = amortized_likelihood

    def call(self, input_dict, **kwargs):
        """Performs a forward pass through both amortizers.

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing the following mandatory keys:
            `posterior_inputs`  - The input dictionary for the amortized posterior
            `likelihood_inputs` - The input dictionary for the amortized likelihood

        Returns
        -------
        (post_out, lik_out) : tuple
            The outputs of the posterior and likelihood networks given input variables.
        """

        post_out = self.amortized_posterior(input_dict["posterior_inputs"], **kwargs)
        lik_out = self.amortized_likelihood(input_dict["likelihood_inputs"], **kwargs)
        return post_out, lik_out

    def compute_loss(self, input_dict, **kwargs):
        """Computes the loss of the join amortizer by summing the corresponding amortized posterior
        and likelihood losses.

        Parameters
        ----------
        input_dict   : dict
            Nested input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged::
            `posterior_inputs`  - The input dictionary for the amortized posterior
            `likelihood_inputs` - The input dictionary for the amortized likelihood

        Returns
        -------
        total_losses : dict
            A dictionary with keys `Post.Loss` and `Lik.Loss` containing the individual losses for the
            two amortizers.
        """

        loss_post = self.amortized_posterior.compute_loss(input_dict[DEFAULT_KEYS["posterior_inputs"]], **kwargs)
        loss_lik = self.amortized_likelihood.compute_loss(input_dict[DEFAULT_KEYS["likelihood_inputs"]], **kwargs)
        return {"Post.Loss": loss_post, "Lik.Loss": loss_lik}

    def log_likelihood(self, input_dict, to_numpy=True, **kwargs):
        """Calculates the approximate log-likelihood of data given conditional variables via
        the change-of-variable formula for conditional normalizing flows.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged:

            `observables` - the variables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that are directly passed to the inference network

            OR a nested dictionary with key `likelihood_inputs` containing the above input dictionary
        to_numpy   : bool, optional, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        log_lik     : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        if input_dict.get(DEFAULT_KEYS["likelihood_inputs"]) is not None:
            return self.amortized_likelihood.log_likelihood(
                input_dict[DEFAULT_KEYS["likelihood_inputs"]], to_numpy=to_numpy, **kwargs
            )
        return self.amortized_likelihood.log_likelihood(input_dict, to_numpy=to_numpy, **kwargs)

    def log_posterior(self, input_dict, to_numpy=True, **kwargs):
        """Calculates the approximate log-posterior of targets given conditional variables via
        the change-of-variable formula for conditional normalizing flows.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged:

            `parameters`         - the latent generative model parameters over which a condition density is learned
            `summary_conditions` - the conditioning variables that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the inference network

            OR a nested dictionary with key `posterior_inputs` containing the above input dictionary

        Returns
        -------
        log_post    : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        if input_dict.get(DEFAULT_KEYS["posterior_inputs"]) is not None:
            return self.amortized_posterior.log_posterior(
                input_dict[DEFAULT_KEYS["posterior_inputs"]], to_numpy=to_numpy, **kwargs
            )
        return self.amortized_posterior.log_posterior(input_dict, to_numpy=to_numpy, **kwargs)

    def log_prob(self, input_dict, to_numpy=True, **kwargs):
        """Identical to calling separate `log_likelihood()` and `log_posterior()`.

        Returns
        -------
        out_dict : dict with keys `log_posterior` and `log_likelihood` corresponding
        to the computed log_pdfs of the approximate posterior and likelihood.
        """

        log_post = self.log_posterior(input_dict, to_numpy=to_numpy, **kwargs)
        log_lik = self.log_likelihood(input_dict, to_numpy=to_numpy, **kwargs)
        out_dict = {"log_posterior": log_post, "log_likelihood": log_lik}
        return out_dict

    def sample_data(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """Generates `n_samples` random draws from the surrogate likelihood given input conditions.

        Parameters
        ----------

        input_dict   : dict
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged:

            `conditions` - the conditioning variables that the directly passed to the inference network

            OR a nested dictionary with key `likelihood_inputs` containing the above input dictionary
        n_samples    : int
            The number of posterior samples to obtain from the approximate posterior
        to_numpy     : bool, optional, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        lik_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, None)
            Simulated observables from the surrogate likelihood.
        """

        if input_dict.get(DEFAULT_KEYS["likelihood_inputs"]) is not None:
            return self.amortized_likelihood.sample(
                input_dict[DEFAULT_KEYS["likelihood_inputs"]], n_samples, to_numpy=to_numpy, **kwargs
            )
        return self.amortized_likelihood.sample(input_dict, n_samples, to_numpy=to_numpy, **kwargs)

    def sample_parameters(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """Generates random draws from the approximate posterior given conditonal variables.

        Parameters
        ----------
        input_dict   : dict
            Input dictionary containing the following mandatory keys, if DEFAULT KEYS unchanged:

            `summary_conditions` : the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  : the conditioning variables that the directly passed to the inference network

            OR a nested dictionary with key `posterior_inputs` containing the above input dictionary
        n_samples    : int
            The number of posterior samples to obtain from the approximate posterior
        to_numpy     : bool, optional, default: True
            Boolean flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_datasets, n_samples, n_params)
            the sampled parameters per data set
        """

        if input_dict.get(DEFAULT_KEYS["posterior_inputs"]) is not None:
            return self.amortized_posterior.sample(
                input_dict[DEFAULT_KEYS["posterior_inputs"]], n_samples, to_numpy=to_numpy, **kwargs
            )
        return self.amortized_posterior.sample(input_dict, n_samples, to_numpy=to_numpy, **kwargs)

    def sample(self, input_dict, n_post_samples, n_lik_samples, to_numpy=True, **kwargs):
        """Identical to calling `sample_parameters()` and `sample_data()` separately.

        Returns
        -------
        out_dict : dict with keys `posterior_samples` and `likelihood_samples` corresponding
        to the `n_samples` from the approximate posterior and likelihood, respectively
        """

        post_samples = self.sample_parameters(input_dict, n_post_samples, to_numpy=to_numpy, **kwargs)
        lik_samples = self.sample_data(input_dict, n_lik_samples, to_numpy=to_numpy, **kwargs)
        out_dict = {"posterior_samples": post_samples, "likelihood_samples": lik_samples}
        return out_dict


class AmortizedModelComparison(tf.keras.Model):
    """An interface to connect an evidential network for Bayesian model comparison with an optional summary network,
    as described in the original paper on evidential neural networks for model comparison according to [1, 2]:

    [1] Radev, S. T., D'Alessandro, M., Mertens, U. K., Voss, A., Köthe, U., & Bürkner, P. C. (2021).
    Amortized bayesian model comparison with evidential deep learning.
    IEEE Transactions on Neural Networks and Learning Systems.

    [2] Elsemüller, L., Schnuerch, M., Bürkner, P. C., & Radev, S. T. (2023).
    A Deep Learning Method for Comparing Bayesian Hierarchical Models.
    arXiv preprint arXiv:2301.11873.

    Note: the original paper [1] does not distinguish between the summary and the evidential networks, but
    treats them as a whole, with the appropriate architecture dictated by the model application. For the
    sake of consistency and modularity, the BayesFlow library separates the two constructs.
    """

    def __init__(self, inference_net, summary_net=None, loss_fun=None):
        """Initializes a composite neural architecture for amortized bayesian model comparison.

        Parameters
        ----------
        inference_net     : tf.keras.Model
            A neural network which outputs model evidences.
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network
        loss_fun          : callable or None, optional, default: None
            The loss function which accepts the outputs of the amortizer. If None, the loss will be the log-loss.

        Important
        ----------
        - If no ``summary_net`` is provided, then the output dictionary of your generative model should not contain
        any `sumamry_conditions`, i.e., ``summary_conditions`` should be set to None, otherwise these will be ignored.

        - If no custom ``loss_fun`` is provided, the loss function will be the log loss for the means of a Dirichlet
        distribution or softmax outputs.
        """

        super().__init__()

        self.inference_net = inference_net
        self.summary_net = summary_net
        self.loss = self._determine_loss(loss_fun)
        self.num_models = self.inference_net.num_models

    def call(self, input_dict, return_summary=False, **kwargs):
        """Performs a forward pass through both networks.

        Parameters
        ----------
        input_dict     : dict
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged
            `summary_conditions` - the conditioning variables that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the evidential network
            `model_indices`      - the ground-truth, one-hot encoded model indices sampled from the model prior
        return_summary : bool, optional, default: False
            Indicates whether the summary network outputs are returned along the estimated evidences.

        Returns
        -------
        net_out : tf.Tensor of shape (batch_size, num_models) or tuple of (net_out (batch_size, num_models),
                  summary_out (batch_size, summary_dim)), the latter being the summary network outputs, if
                  ``return_summary is True``.
        """

        summary_out, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(DEFAULT_KEYS["direct_conditions"]),
            **kwargs,
        )

        net_out = self.inference_net(full_cond, **kwargs)

        if not return_summary:
            return net_out
        return net_out, summary_out

    def posterior_probs(self, input_dict, to_numpy=True, **kwargs):
        """Compute posterior model probabilities (PMPs) given a dictionary with observed or
        simulated data.

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing at least one of the following mandatory keys, if DEFAULT_KEYS unchanged
            `summary_conditions` - the conditioning variables that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the evidential network
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the PMPs a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        out       : tf.Tensor of shape (batch_size, ..., num_models)
            The approximated PMPs
        """

        _, full_cond = self._compute_summary_condition(
            input_dict.get(DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(DEFAULT_KEYS["direct_conditions"]),
            **kwargs,
        )
        pmps = self.inference_net(full_cond, **kwargs)
        if to_numpy:
            return pmps.numpy()
        return pmps

    def compute_loss(self, input_dict, **kwargs):
        """Computes the loss of the amortized model comparison instance.

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged::
            `summary_conditions` - the conditioning variables that are first passed through a summary network
            `direct_conditions`  - the conditioning variables that the directly passed to the evidence network
            `model_indices`      - the ground-truth, one-hot encoded model indices sampled from the model prior

        Returns
        -------
        loss  : tf.Tensor of shape (1,) - the total computed loss given input variables
        """

        preds = self(input_dict, **kwargs)
        loss = self.loss(input_dict[DEFAULT_KEYS["model_indices"]], preds)
        return loss

    def _compute_summary_condition(self, summary_conditions, direct_conditions, **kwargs):
        """Helper method to determines how to concatenate the provided conditions."""

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sum_condition = self.summary_net(summary_conditions, **kwargs)
        else:
            sum_condition = None

        # Concatenate learnable summaries with fixed summaries, this
        if sum_condition is not None and direct_conditions is not None:
            full_cond = tf.concat([sum_condition, direct_conditions], axis=-1)
        elif sum_condition is not None:
            full_cond = sum_condition
        elif direct_conditions is not None:
            full_cond = direct_conditions
        else:
            raise SummaryStatsError("Could not concatenarte or determine conditioning inputs...")
        return sum_condition, full_cond

    def _determine_loss(self, loss_fun):
        """Helper method to determine loss function to use."""

        if loss_fun is None:
            return partial(log_loss, evidential=isinstance(self.inference_net, EvidentialNetwork))
        elif callable(loss_fun):
            return loss_fun
        else:
            raise ConfigurationError(
                "Loss function is neither default (`None`) not callable. Please provide a valid loss function!"
            )


class TwoLevelAmortizedPosterior(tf.keras.Model, AmortizedTarget):
    """An interface for estimating arbitrary two level hierarchical Bayesian models."""

    def __init__(self, local_amortizer, global_amortizer, summary_net=None, **kwargs):
        """Creates an wrapper for estimating two-level hierarchical Bayesian models.

        Parameters
        ----------
        local_amortizer   : bayesflow.amortizers.AmortizedPosterior
            A posterior amortizer without a summary network which will estimate
            the full conditional of the (varying numbers of) local parameter vectors.
        global_amortizer  : bayesflow.amortizers.AmortizedPosterior
            A posterior amortizer without a summary network which will estimate the joint
            posterior of hyperparameters and optional shared parameters given a representation
            of an entire hierarchical data set. If both hyper- and shared parameters are present,
            the first dimensions correspond to the hyperparameters and the remaining ones correspond
            to the shared parameters.
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network to compress non-vector data structures.
        **kwargs          : dict, optional, default: {}
            Additional keyword arguments passed to the ``__init__`` method of a ``tf.keras.Model`` instance.
        """

        super().__init__(**kwargs)

        self.local_amortizer = local_amortizer
        self.global_amortizer = global_amortizer
        self.summary_net = summary_net

    def call(self, input_dict, **kwargs):
        """Forward pass through the hierarchical amortized posterior."""

        local_summaries, global_summaries = self._compute_condition(input_dict, **kwargs)
        local_inputs, global_inputs = self._prepare_inputs(input_dict, local_summaries, global_summaries)
        local_out = self.local_amortizer(local_inputs, **kwargs)
        global_out = self.global_amortizer(global_inputs, **kwargs)
        return local_out, global_out

    def compute_loss(self, input_dict, **kwargs):
        """Compute loss of all amortizers."""

        local_summaries, global_summaries = self._compute_condition(input_dict, **kwargs)
        local_inputs, global_inputs = self._prepare_inputs(input_dict, local_summaries, global_summaries)
        local_loss = self.local_amortizer.compute_loss(local_inputs, **kwargs)
        global_loss = self.global_amortizer.compute_loss(global_inputs, **kwargs)
        return {"Local.Loss": local_loss, "Global.Loss": global_loss}

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """Obtains samples from the joint hierarchical posterior given observations.

        Important: Currently works only for single hierarchical data sets!

        Parameters
        ----------
        input_dict   : dict
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged:
            `summary_conditions` - the hierarchical data set (to be embedded by the summary net)
            As well as optional keys:
            `direct_local_conditions`  - (Context) variables used to condition the local posterior
            `direct_global_conditions` - (Context) variables used to condition the global posterior
        n_samples    : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        to_numpy     : bool, optional, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`
        **kwargs     : dict, optional, default: {}
            Additional keyword arguments passed to the summary network as the amortizers

        Returns
        -------
        samples_dict : dict
            A dictionary with keys `global_samples` and `local_samples`
            Local samples will hold an array-like of shape (num_replicas, num_samples, num_local)
            and local samples will hold an array-like of shape (num_samples, num_hyper + num_shared),
            if optional shared patameters are present, otherwise (num_samples, num_hyper),
        """

        # Returned shapes will be :
        # local_summaries.shape = (1, num_groups, summary_dim_local)
        # global_summaries.shape = (1, summary_dim_global)
        local_summaries, global_summaries = self._get_local_global(input_dict, **kwargs)
        num_groups = local_summaries.shape[1]

        if local_summaries.shape[0] != 1 or global_summaries.shape[0] != 1:
            raise NotImplementedError("Method currently supports only single hierarchical data sets!")

        # Obtain samples from p(global | all_data)
        inp_global = {DEFAULT_KEYS["direct_conditions"]: global_summaries}

        # New, shape will be (n_samples, num_globals)
        global_samples = self.global_amortizer.sample(inp_global, n_samples, **kwargs, to_numpy=False)

        # Repeat local conditions for n_samples
        # New shape -> (num_groups, n_samples, summary_dim_local)
        local_summaries = tf.stack([tf.squeeze(local_summaries, axis=0)] * n_samples, axis=1)

        # Repeat global samples for num_groups
        # New shape -> (num_groups, n_samples, num_globals)
        global_samples_rep = tf.stack([global_samples] * num_groups, axis=0)

        # Concatenate local summaries with global samples
        # New shape -> (num_groups, num_samples, summary_dim_local + num_globals)
        local_summaries = tf.concat([local_summaries, global_samples_rep], axis=-1)

        # Obtain samples from p(local_i | data_i, global_i)
        inp_local = {DEFAULT_KEYS["direct_conditions"]: local_summaries}
        local_samples = self.local_amortizer.sample(inp_local, n_samples, to_numpy=False, **kwargs)

        if to_numpy:
            global_samples = global_samples.numpy()
            local_samples = local_samples.numpy()

        return {"global_samples": global_samples, "local_samples": local_samples}

    def log_prob(self, input_dict):
        """Compute normalized log density."""

        raise NotImplementedError

    def _prepare_inputs(self, input_dict, local_summaries, global_summaries):
        """Prepare input dictionaries for both amortizers."""

        # Prepare inputs for local amortizer
        local_inputs = {"direct_conditions": local_summaries, "parameters": input_dict["local_parameters"]}

        # Prepare inputs for global amortizer
        _parameters = input_dict["hyper_parameters"]
        if input_dict.get("shared_parameters") is not None:
            _parameters = tf.concat([_parameters, input_dict.get("shared_parameters")], axis=-1)
        global_inputs = {"direct_conditions": global_summaries, "parameters": _parameters}
        return local_inputs, global_inputs

    def _compute_condition(self, input_dict, **kwargs):
        """Determines conditionining variables for both amortizers."""

        # Obtain needed summaries
        local_summaries, global_summaries = self._get_local_global(input_dict, **kwargs)

        # At this point, add globals as conditions
        num_locals = tf.shape(local_summaries)[1]

        # Add hyper parameters as conditions:
        # p(local_n | data_n, hyper)
        if input_dict.get("hyper_parameters") is not None:
            _params = input_dict.get("hyper_parameters")
            _params = tf.expand_dims(_params, 1)
            _conds = tf.tile(_params, [1, num_locals, 1])
            local_summaries = tf.concat([local_summaries, _conds], axis=-1)

        # Add shared parameters as conditions:
        # p(local_n | data_n, hyper, shared)
        if input_dict.get("shared_parameters") is not None:
            _params = input_dict.get("shared_parameters")
            _params = tf.expand_dims(_params, 1)
            _conds = tf.tile(_params, [1, num_locals, 1])
            local_summaries = tf.concat([local_summaries, _conds], axis=-1)
        return local_summaries, global_summaries

    def _get_local_global(self, input_dict, **kwargs):
        """Helper function to obtain local and global condition tensors."""

        # Obtain summary conditions
        if self.summary_net is not None:
            local_summaries, global_summaries = self.summary_net(
                input_dict["summary_conditions"], return_all=True, **kwargs
            )
            if input_dict.get("direct_local_conditions") is not None:
                local_summaries = tf.concat([local_summaries, input_dict.get("direct_local_conditions")], axis=-1)
            if input_dict.get("direct_global_conditions") is not None:
                global_summaries = tf.concat([global_summaries, input_dict.get("direct_global_conditions")], axis=-1)
        # If no summary net provided, assume direct conditions exist or fail
        else:
            local_summaries = input_dict.get("direct_local_conditions")
            global_summaries = input_dict.get("direct_global_conditions")
        return local_summaries, global_summaries


class SingleModelAmortizer(AmortizedPosterior):
    """Deprecated class for amortizer posterior estimation."""

    def __init_subclass__(cls, **kwargs):
        warn(f"{cls.__name__} will be deprecated. Use `AmortizedPosterior` instead.", DeprecationWarning, stacklevel=2)
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        warn(
            f"{self.__class__.__name__} will be deprecated. Use `AmortizedPosterior` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
