
import keras
from keras import layers
from keras.saving import register_keras_serializable

from bayesflow.experimental.types import Tensor
from .hidden_block import ConfigurableHiddenBlock


@register_keras_serializable(package="bayesflow.networks.resnet")
class ResNet(keras.layers.Layer):
    """
    Implements a simple configurable MLP with optional residual connections and dropout.

    If used in conjunction with a coupling net, a diffusion model, or a flow matching model, it assumes
    that the input and conditions are already concatenated (i.e., this is a single-input model).
    """

    def __init__(
        self,
        num_hidden: int = 2,
        hidden_dim: int = 256,
        activation: str = "gelu",
        kernel_initializer: str = "he_uniform",
        residual: bool = True,
        dropout_rate: float = 0.05,
        spectral_normalization: bool = False,
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
        spectral_normalization    : bool, optional, default: False
            Use spectral normalization for the network weights, which can make
            the learned function smoother and hence more robust to perturbations.
        dropout_rate     : float, optional, default: 0.05
            Dropout rate for the hidden layers in the internal layers.
        """

        super().__init__(**kwargs)

        self.res_blocks = keras.Sequential()
        projector = layers.Dense(
            units=hidden_dim,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            bias_regularizer=bias_regularizer,
        )
        if spectral_normalization:
            projector = layers.SpectralNormalization(projector)
        self.res_blocks.add(projector)
        self.res_blocks.add(layers.Dropout(dropout_rate))

        for _ in range(num_hidden):
            self.res_blocks.add(
                ConfigurableHiddenBlock(
                    units=hidden_dim,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    residual=residual,
                    dropout_rate=dropout_rate,
                    spectral_normalization=spectral_normalization
                )
            )

    def call(self, inputs: Tensor, **kwargs):
        return self.res_blocks(inputs, training=kwargs.get("training", False))

