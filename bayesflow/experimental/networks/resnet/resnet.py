
from typing import Sequence

import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Tensor


@register_keras_serializable(package="bayesflow.networks")
class ResNet(keras.Layer):
    """
    Implements a simple, fully-connected residual network.
    Input tensors are linearly mapped to the width of the network to ensure shape-compatibility.
    """
    def __init__(self, input_layer: keras.Layer, hidden_layers: Sequence[keras.Layer], output_layer: keras.Layer, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = input_layer
        self.hidden_layers = list(hidden_layers)
        self.output_layer = output_layer

    @classmethod
    def new(cls, depth: int = 4, width: int = 256, activation: str = "relu", **kwargs) -> "ResNet":
        input_layer = keras.layers.Dense(width)
        hidden_layers = [keras.layers.Dense(width, activation=activation) for _ in range(depth)]
        output_layer = keras.layers.Dense(width)

        return cls(input_layer, hidden_layers, output_layer, **kwargs)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ResNet":
        input_layer = deserialize_keras_object(config.pop("input_layer"))
        hidden_layers = deserialize_keras_object(config.pop("hidden_layers"))
        output_layer = deserialize_keras_object(config.pop("output_layer"))

        return cls(input_layer, hidden_layers, output_layer, **config)

    def get_config(self) -> dict:
        base_config = super().get_config()

        config = {
            "input_layer": serialize_keras_object(self.input_layer),
            "hidden_layers": serialize_keras_object(self.hidden_layers),
            "output_layer": serialize_keras_object(self.output_layer),
        }

        return base_config | config

    def build(self, input_shape):
        self.call(keras.KerasTensor(input_shape))

    def build_output(self, output_shape):
        match self.output_layer:
            case keras.layers.Dense() as dense:
                dense.units = output_shape[-1]
            case other:
                raise NotImplementedError(f"Cannot build output for layer {other!r}")

    def call(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = x + layer(x)
        x = self.output_layer(x)

        return x
