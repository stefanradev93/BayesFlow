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
import aesara.tensor as at
from aesara.graph.op import Op

from bayesflow.default_settings import DEFAULT_KEYS
from bayesflow.amortized_inference import AmortizedLikelihood


class MCMCSurrogateLikelihood:
    """ An interface to provide likelihood and gradient of a pre-trained
    amortized likelihood.
    """
    @tf.function
    def __init__(self, amortized_likelihood, configurator=None, likelihood_postprocessor=None,
                 grad_postprocessor=None):
        """
        Parameters
        ----------
        amortized_likelihood : bayesflow.amortized_inference.AmortizedLikelihood
            A pre-trained (invertible) inference network which processes the outputs of the generative model.
        configurator  : callable
            A function that takes the input to the `log_likelihood` and `log_likelihood_grad`
            calls and converts them to a dictionary containing the following mandatory keys,
            if DEFAULT_KEYS unchanged: 
                `observables` - the variables over which a condition density is learned (i.e., the observables)
                `conditions`  - the conditioning variables that the directly passed to the inference network
            default: Return the first parameter - has to be a dicitionary with the mentioned characteristics
        likelihood_postprocessor  : callable
            A function that takes the likelihood for each observable as an input. Can be used for aggregation
            default: sum all likelihoods and return a single value
        grad_postprocessor  : callable
            A function that takes the gradient for each value in `conditions` as returned by the preprocessor
            default: Leave the values unchanged
            
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
        
        Parameters:
        args, kwargs: Arguments passed to the `configurator`
        """
        # print(args, kwargs)
        input_dict = self.configurator(*args, **kwargs)
        return self.likelihood_postprocessor(
            self.amortized_likelihood.log_likelihood(input_dict, to_numpy=False, **kwargs)
        )


    def log_likelihood_grad(self, *args, **kwargs):
        """ Calculates the gradient with respect to every parameter in `conditions`.
        
        Parameters:
        args, kwargs: Arguments passed to the `configurator`
        """
        input_dict = self.configurator(*args, **kwargs)
        observables = tf.constant(np.float32(input_dict[DEFAULT_KEYS['observables']]), dtype=np.float32)
        conditions = tf.Variable(np.float32(input_dict[DEFAULT_KEYS['conditions']]), dtype=np.float32)
        # print(observables, conditions)
        return self.grad_postprocessor(self._log_likelihood_grad({
            DEFAULT_KEYS['observables']: observables,
            DEFAULT_KEYS['conditions']: conditions
        }, **kwargs))

    
    @tf.function
    def _log_likelihood_grad(self, input_dict, **kwargs):
        """ Calculates the gradient with respect to every parameter in `conditions`.
        
        Parameters:
        input_dict : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS unchanged: 
                `observables` - tf.constant: the variables over which a condition density is learned (i.e., the observables)
                `conditions`  - tf.Variable: the conditioning variables that the directly passed to the inference network
        """
        with tf.GradientTape() as t:
            # obs = input_dict[DEFAULT_KEYS['observables']]
            # conditions = input_dict[DEFAULT_KEYS['conditions']]
            log_lik = tf.reduce_sum(self.amortized_likelihood.log_likelihood(
                input_dict, to_numpy=False, **kwargs
            ))
        return t.gradient(log_lik, {'p': input_dict[DEFAULT_KEYS['conditions']]})['p']


# Custom log-likelihood operator, based on 
# https://www.pymc.io/projects/examples/en/latest/case_studies
# /blackbox_external_likelihood_numpy.html#aesara-op-with-grad
class _LogLikeGrad(at.Op):

    """
    This Op will execute the `loglikegrad` function, supplying the gradient
    for the custom likelihood function.
    """

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, loglikegrad, observables):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglikegrad: callable
            The function that provides the gradient of the log-likelihood
            function
        observables:
            The "observed" data that the log-likelihood function takes in
        """

        # add inputs as class attributes
        self.observables = observables
        self.loglikegrad = loglikegrad

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # calculate gradients
        outputs[0][0] = np.float64(self.loglikegrad(self.observables, theta).numpy())


# define a aesara Op for our likelihood function
class PyMCSurrogateLikelihood(at.Op, MCMCSurrogateLikelihood):

    itypes = [at.dvector]  # expects a vector of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, amortized_likelihood, observables, configurator=None, likelihood_postprocessor=None,
                 grad_postprocessor=None):
        """
        A custom likelihood function, to be used with pymc.Potential

        Parameters
        ----------
        amortized_likelihood : bayesflow.amortized_inference.AmortizedLikelihood
            A pre-trained (invertible) inference network which processes the outputs of the generative model.
        observables:
            The "observed" data that will be passed to the configurator
        configurator  : callable
            A function that takes the input to the lpdf and grad calls and converts the to a dictionary
            containing the following mandatory keys, if DEFAULT_KEYS unchanged: 
            `observables` - the variables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that the directly passed to the inference network
            default: Return the first parameter - has to be a dicitionary with the mentioned characteristics
        likelihood_postprocessor  : callable
            A function that takes the likelihood for each observable, can be used for aggregation
            default: sum all likelihoods and return a single value
        grad_postprocessor  : callable
            A function that takes the gradient for each value in `conditions` as returned by the preprocessor
            default: Leave the values unchanged
        """
        MCMCSurrogateLikelihood.__init__(
            self,
            amortized_likelihood=amortized_likelihood,
            configurator=configurator,
            likelihood_postprocessor=likelihood_postprocessor,
            grad_postprocessor=grad_postprocessor
        )

        # add inputs as class attributes
        self.observables = observables

        # initialise the gradient Op
        self.logpgrad = _LogLikeGrad(self.log_likelihood_grad, self.observables)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain the parameters

        # call the log-likelihood function
        logl = self.log_likelihood(self.observables, np.float32(theta))

        outputs[0][0] = np.array(logl, dtype=np.float64)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters

        return [g[0] * self.logpgrad(theta)]
