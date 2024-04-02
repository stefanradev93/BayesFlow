
import keras


class InferenceNetwork(keras.Layer):
    def __init__(self, configurator: Configurator = Configurator(), **kwargs):
        super().__init__(**kwargs)
        self.configurator = configurator

    def call(self, data, conditions, summaries=None):
        raise NotImplementedError

    def compute_loss(self, data, conditions, summaries=None):
        raise NotImplementedError
