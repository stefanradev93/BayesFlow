import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.experimental.types import Tensor


@register_keras_serializable(package="bayesflow.networks")
class ResNet(keras.Layer):
    """
    Implements a simple, fully-connected residual network.
    Input tensors are linearly mapped to the correct dimension.
    """
    def __init__(self, depth: int = 6, width: int = 256, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)

        self.input_layer = keras.layers.Dense(width)
        self.hidden_layers = [keras.layers.Dense(width, activation=activation) for _ in range(depth - 1)]
        self.output_layer = keras.layers.Dense(width, activation=activation)

    def call(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = x + layer(x)
        x = x + self.output_layer(x)

        return x
