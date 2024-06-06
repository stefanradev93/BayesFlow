import keras
from keras.saving import register_keras_serializable
from bayesflow.experimental.types import Tensor

@register_keras_serializable(package="bayesflow.networks")
class SkipGRU(keras.Model):
    # TODO
    pass