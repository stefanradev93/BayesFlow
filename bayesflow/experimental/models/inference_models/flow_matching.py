
import keras

from bayesflow.experimental import utils

from .inference_network import InferenceNetwork


class FlowMatching(keras.Model):
    def __init__(self, network: keras.Layer, **kwargs):
        super().__init__(**kwargs)
        self.network = network

    def call(self, parameters, conditions=None, summaries=None):

        if c is None:
            shape = keras.ops.shape(x1)
            shape[1] = 0
            c = keras.ops.zeros(shape, axis=1)

        x0 = keras.ops.zeros_like(x1)
        t = keras.random.uniform(keras.ops.shape(x1)[0])
        t = utils.expand_right(t, keras.ops.ndim(x1) - 1)

        x = t * x1 + (1 - t) * x0

        return self.network(x, t, c)

    def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):

        return keras.losses.mean_squared_error(y, y_pred)


class FlowMatchingWrapper(InferenceNetwork):
    def __init__(self, network: keras.Layer, **kwargs):
        super().__init__(**kwargs)
        self.fm = FlowMatching(network, **kwargs)

