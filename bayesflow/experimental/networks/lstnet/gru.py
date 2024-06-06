import keras
from keras.saving import register_keras_serializable
from bayesflow.experimental.types import Tensor

@register_keras_serializable(package="bayesflow.networks")
class GRU(keras.Model):
    # TODO
    pass