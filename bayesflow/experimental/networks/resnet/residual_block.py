
import keras
from keras.saving import (
    register_keras_serializable,
)

from .hidden_block import ConfigurableHiddenBlock


@register_keras_serializable(package="bayesflow.networks.resnet")
class ConditionalResidualBlock(keras.layers.Layer):
    """
    Implements a simple configurable MLP with optional residual connections and dropout.

    If used in conjunction with a coupling net, a diffusion model, or a flow matching model, it assumes
    that the input and conditions are already concatenated (i.e., this is a single-input model).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        activation: str | callable = "gelu",
        residual: bool = True,
        spectral_norm: bool = False,
        dropout_rate: float = 0.05,
        **kwargs
    ):
        """
        Creates an instance of a flexible and simple MLP with optional residual connections and dropout.

        Parameters:
        -----------
        hidden_dim       : int, optional, default: 256
            The dimensionality of the hidden layers
        num_hidden       : int, optional, default: 2
            The number of hidden layers (minimum: 1)
        activation       : string, optional, default: 'gelu'
            The activation function of the dense layers
        residual         : bool, optional, default: True
            Use residual connections in the internal layers.
        spectral_norm    : bool, optional, default: False
            Use spectral normalization for the network weights, which can make
            the learned function smoother and hence more robust to perturbations.
        dropout_rate     : float, optional, default: 0.05
            Dropout rate for the hidden layers in the internal layers.
        """

        super().__init__(**kwargs)

        self.res_blocks = keras.Sequential(
            [keras.layers.Dense(hidden_dim, activation=activation), keras.layers.Dropout(dropout_rate)]
        )
        for _ in range(num_hidden):
            self.res_blocks.add(
                ConfigurableHiddenBlock(
                    num_units=hidden_dim,
                    activation=activation,
                    residual=residual,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm
                )
            )

    def call(self, inputs, training=False):
        return self.res_blocks(inputs, training=training)
