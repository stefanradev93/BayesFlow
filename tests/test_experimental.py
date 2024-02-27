
import math

import tensorflow as tf
import tensorflow_probability.python.distributions as D

from bayesflow.experimental.backend_agnostic import GenerativeModel
from bayesflow.experimental.backend_agnostic import SampleContextsMixin, SampleParametersMixin, SampleObservationsMixin

context_priors = {
    "r": D.Normal(0.1, 0.01),
    "alpha": D.Uniform(-math.pi / 2, math.pi / 2),
}


parameter_priors = {
    "theta1": D.Uniform(-1, 1),
    "theta2": D.Uniform(-1, 1),
}

@Simulator
def simulator(batch_shape, /, *, parameters, contexts=None):
    theta1, theta2 = parameters["theta1"], parameters["theta2"]
    r, alpha = contexts["r"], contexts["alpha"]

    x1 = -tf.abs(theta1 + theta2) / tf.sqrt(2) + r * tf.cos(alpha) + 0.25
    x2 = (-theta1 + theta2) / tf.sqrt(2) + r * tf.sin(alpha)

    return {"x1": x1, "x2": x2}




class ContextPrior(SampleContextsMixin):
    def sample_contexts(self, batch_shape, /):
        r = D.Normal(0.1, 0.01).sample(batch_shape)
        alpha = D.Uniform(-tf.pi / 2, tf.pi / 2).sample(batch_shape)

        return {"r": r, "alpha": alpha}


class ParameterPrior(SampleParametersMixin):
    def sample_parameters(self, batch_shape, /, *, contexts=None):
        theta1 = D.Uniform(-1, 1).sample(batch_shape)
        theta2 = D.Uniform(-1, 1).sample(batch_shape)

        return {"theta1": theta1, "theta2": theta2}


class Simulator(SampleObservationsMixin):
    def sample_observations(self, batch_shape, /, *, parameters, contexts=None):
        theta1, theta2 = parameters["theta1"], parameters["theta2"]
        r, alpha = contexts["r"], contexts["alpha"]

        x1 = -tf.abs(theta1 + theta2) / tf.sqrt(2) + r * tf.cos(alpha) + 0.25
        x2 = (-theta1 + theta2) / tf.sqrt(2) + r * tf.sin(alpha)

        return {"x1": x1, "x2": x2}


def test_experimental():
    generative_model = GenerativeModel(ParameterPrior(), Simulator(), ContextPrior())
    print(generative_model.sample((10, 2)))
