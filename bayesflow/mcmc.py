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

import aesara.tensor as at
import numpy as np
import tensorflow as tf

from bayesflow.default_settings import DEFAULT_KEYS


class MCMCSurrogateLikelihood:
    """An interface to provide likelihood evaluation and gradient estimation of a pre-trained
    ``AmortizedLikelihood`` instance, which can be used in tandem with (HMC)-MCMC, as implemented,
    for instance, in ``PyMC3``.
    """

    @tf.function
    def __init__(self, amortized_likelihood, configurator=None, likelihood_postprocessor=None, grad_postprocessor=None):
        """Creates in instance of the surrogate likelihood using a pre-trained ``AmortizedLikelihood`` instance.

        Parameters
        ----------
        amortized_likelihood       : bayesflow.amortized_inference.AmortizedLikelihood
            A pre-trained (invertible) inference network which processes the outputs of the generative model.
        configurator               : callable, optional, default: None
            A function that takes the input to the ``log_likelihood`` and ``log_likelihood_grad``
            calls and converts them to a dictionary containing the following mandatory keys,
            if DEFAULT_KEYS unchanged:
                ``observables`` - the variables over which a condition density is learned (i.e., the observables)
                ``conditions``  - the conditioning variables that the directly passed to the inference network
            default: Return the first parameter - has to be a dicitionary with the mentioned characteristics
        likelihood_postprocessor  : callable, optional, default: None
            A function that takes the likelihood for each observable as an input. Can be used for aggregation
            default: sum all likelihood values and return a single value.
        grad_postprocessor        : callable, optional, default: None
            A function that takes the gradient for each value in ``conditions`` as returned by the preprocessor
            default: Leave the values unchanged.
        """

        self.amortized_likelihood = amortized_likelihood
        self.configurator = configurator
        if self.configurator is None:
            self.configurator = self._default_configurator
        self.likelihood_postprocessor = likelihood_postprocessor
        if self.likelihood_postprocessor is None:
            self.likelihood_postprocessor = self._default_likelihood_postprocessor
        self.grad_postprocessor = grad_postprocessor
        if self.grad_postprocessor is None:
            self.grad_postprocessor = self._default_grad_postprocessor

    @tf.function
    def _default_configurator(self, input_dict, *args, **kwargs):
        return input_dict

    @tf.function
    def _default_likelihood_postprocessor(self, values):
        return tf.reduce_sum(values)

    @tf.function
    def _default_grad_postprocessor(self, values):
        return values

    @tf.function
    def log_likelihood(self, *args, **kwargs):
        """Calculates the approximate log-likelihood of targets given conditional variables.

        Parameters
        ----------
        The parameters as expected by ``configurator``. For the default configurator,
        the first parameter has to be a dictionary containing the following mandatory keys,
        if DEFAULT_KEYS unchanged:
            ``observables`` - the variables over which a condition density is learned (i.e., the observables)
            ``conditions``  - the conditioning variables that the directly passed to the inference network

        Returns
        -------
        out : np.ndarray
            The output as returned by ``likelihood_postprocessor``. For the default postprocessor,
            this is the total log-likelihood given by the sum of all log-likelihood values.
        """

        input_dict = self.configurator(*args, **kwargs)
        return self.likelihood_postprocessor(
            self.amortized_likelihood.log_likelihood(input_dict, to_numpy=False, **kwargs)
        )

    def log_likelihood_grad(self, *args, **kwargs):
        """Calculates the gradient of the surrogate likelihood with respect to
        every parameter in ``conditions``.

        Parameters
        ----------
        The parameters as expected by ``configurator``. For the default configurator,
        the first parameter has to be a dictionary containing the following mandatory keys,
        if ``DEFAULT_KEYS`` unchanged:
            ``observables`` - the variables over which a condition density is learned (i.e., the observables)
            ``conditions``  - the conditioning variables that the directly passed to the inference network

        Returns
        -------
        out : np.ndarray
            The output as returned by ``grad_postprocessor``. For the default postprocessor,
            this is an array containing the derivative with respect to each value in ``conditions``
            as returned by ``configurator``.
        """

        input_dict = self.configurator(*args, **kwargs)
        observables = tf.constant(np.float32(input_dict[DEFAULT_KEYS["observables"]]), dtype=np.float32)
        conditions = tf.Variable(np.float32(input_dict[DEFAULT_KEYS["conditions"]]), dtype=np.float32)
        return self.grad_postprocessor(
            self._log_likelihood_grad(
                {DEFAULT_KEYS["observables"]: observables, DEFAULT_KEYS["conditions"]: conditions}, **kwargs
            )
        )

    @tf.function
    def _log_likelihood_grad(self, input_dict, **kwargs):
        """Calculates the gradient with respect to every parameter contained in ``input_dict``.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged:
                ``observables`` - tf.constant: the variables over which a condition density is learned (i.e., the observables)
                ``conditions``  - tf.Variable: the conditioning variables that the directly passed to the inference network

        Returns
        -------
        out : tf.Tensor
        """

        with tf.GradientTape() as t:
            log_lik = tf.reduce_sum(self.amortized_likelihood.log_likelihood(input_dict, to_numpy=False, **kwargs))
        return t.gradient(log_lik, {"p": input_dict[DEFAULT_KEYS["conditions"]]})["p"]


class _LogLikGrad(at.Op):
    """Custom log-likelihood operator, based on:
    https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html#aesara-op-with-grad

    This Op will execute the ``log_lik_grad`` function, supplying the gradient
    for the custom surrogate likelihood function.
    """

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, log_lik_grad, observables, default_type=np.float64):
        """Initialize with the gradient function and the observables.

        Parameters
        ----------
        log_lik_grad  : callable
            The object that provides the gradient of the log-likelihood.
        observables   : tf.constant, the shape depends on ``log_lik_grad``.
            The variables over which a condition density is learned (i.e., the observables).
        default_type  : np.dtype, optional, default: np.float64
            The default float type to use for the gradient vectors.
        """

        self.observables = observables
        self.log_lik_grad = log_lik_grad
        self.default_type = default_type

    def perform(self, node, inputs, outputs):
        """Computes gradients with respect to ``inputs`` (corresponding to the parameters of a model).

        Parameters
        ----------
        node      : The symbolic ``aesara.graph.basic.Apply`` node that represents this computation.
        inputs    : Immutable sequence of non-symbolic/numeric inputs. These are the values of each
                    ``Variable`` in ``node.inputs``.
        outputs   : List of mutable single-element lists (do not change the length of these lists).
                    Each sub-list corresponds to value of each ``Variable`` in ``node.outputs``.
                    The primary purpose of this method is to set the values of these sub-lists.
        """

        (theta,) = inputs
        grad = self.log_lik_grad(self.observables, theta)
        # Attempt conversion to numpy, if not already an array
        if type(grad) is not np.ndarray:
            grad = grad.numpy()
        # Store grad in-place
        outputs[0][0] = self.default_type(grad)


class PyMCSurrogateLikelihood(at.Op, MCMCSurrogateLikelihood):

    itypes = [at.dvector]  # expects a vector of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(
        self,
        amortized_likelihood,
        observables,
        configurator=None,
        likelihood_postprocessor=None,
        grad_postprocessor=None,
        default_pymc_type=np.float64,
        default_tf_type=np.float32,
    ):
        """A custom surrogate likelihood function for integration with ``PyMC3``, to be used with pymc.Potential

        Parameters
        ----------
        amortized_likelihood      : bayesflow.amortized_inference.AmortizedLikelihood
            A pre-trained (invertible) inference network which processes the outputs of the generative model.
        observables               :
            The "observed" data that will be passed to the configurator.
            For the default ``configurator``, an np.array of shape (N, x_dim).
        configurator              : callable, optional, default None
            A function that takes the input to the log_likelihood and log_likelihood_grad
            calls and converts it to a dictionary containing the following mandatory keys,
            if DEFAULT_KEYS unchanged:
                ``observables`` - the variables over which a condition density is learned (i.e., the observables)
                ``conditions``  - the conditioning variables that the directly passed to the inference network
            default behavior: convert ``observables`` to shape (1, N, x_dim),
                expand parameters of shape (cond_dim) to shape (1, N, cond_dim)
        likelihood_postprocessor  : callable, optional, default: None
            A function that takes the likelihood for each observable, can be used for aggregation
            default behavior: sum all likelihoods and return a single value
        grad_postprocessor        : callable, optional, default: None
            A function that takes the gradient for each value in ``conditions`` as returned by the preprocessor
            default behavior: Reduce shape from (1, N, cond_dim) to (cond_dim) by summing the corresponding values
        default_pymc_type         : np.dtype, optional, default: np.float64
            The default float type to use for numpy arrays as required by PyMC.
        default_tf_type           : np.dtype, optional, default: np.float32
            The default float type to use for tensorflow tensors.
        """

        MCMCSurrogateLikelihood.__init__(
            self,
            amortized_likelihood=amortized_likelihood,
            configurator=configurator,
            likelihood_postprocessor=likelihood_postprocessor,
            grad_postprocessor=grad_postprocessor,
        )

        self.observables = observables
        self.logpgrad = _LogLikGrad(self.log_likelihood_grad, self.observables, default_type=default_pymc_type)
        self.default_pymc_type = default_pymc_type
        self.default_tf_type = default_tf_type

    @tf.function
    def _default_configurator(self, obs, params):
        return {
            # add axis (corresponds to batch_size=1)
            "observables": obs[tf.newaxis],
            # expand conditions to match number of observables and add axis
            "conditions": tf.tile(params[tf.newaxis, :], [obs.shape[0], 1])[tf.newaxis],
        }

    @tf.function
    def _default_grad_postprocessor(self, grads):
        # remove axis added in configurator and undo expansion by summing
        return tf.reduce_sum(grads[0], axis=0)

    def perform(self, node, inputs, outputs):
        """Computes the log-likelihood of ``inputs`` (typically the parameter vector of a model).

        Parameters
        ----------
        node      : The symbolic ``aesara.graph.basic.Apply`` node that represents this computation.
        inputs    : Immutable sequence of non-symbolic/numeric inputs. These are the values of each
                    ``Variable`` in ``node.inputs``.
        outputs   : List of mutable single-element lists (do not change the length of these lists).
                    Each sub-list corresponds to value of each ``Variable`` in ``node.outputs``.
                    The primary purpose of this method is to set the values of these sub-lists.
        """

        (theta,) = inputs
        logl = self.log_likelihood(self.observables, self.default_tf_type(theta))
        outputs[0][0] = np.array(logl, dtype=self.default_pymc_type)

    def grad(self, inputs, output_grads):
        """Aggregates gradients with respect to ``inputs`` (typically the parameter vector)

        Parameters
        ----------
        inputs        : The input variables.
        output_grads  : The gradients of the output variables.

        Returns
        -------
        grads         : The gradients with respect to each ``Variable`` in ``inputs``.
        """

        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - output_grads[0] is a vector of parameter values
        (theta,) = inputs
        grads = [output_grads[0] * self.logpgrad(theta)]
        return grads
