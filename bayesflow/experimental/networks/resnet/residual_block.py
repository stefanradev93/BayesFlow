
import keras

from .hidden_block import ConfigurableHiddenBlock


class ConditionalResidualBlock(keras.layers.Layer):
    """
    Implements a simple configurable MLP with optional residual connections and dropout.

    If used in conjunction with a coupling net, a diffusion model, or a flow matching model, it assumes
    that the input and conditions are already concatenated (i.e., this is a single-input model).
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim=512,
        num_hidden=2,
        activation="gelu",
        residual=True,
        spectral_norm=False,
        dropout_rate=0.05,
        zero_output_init=True,
        **kwargs
    ):
        """
        Creates an instance of a flexible and simple MLP with optional residual connections and dropout.

        Parameters:
        -----------
        output_dim       : int
            The output dimensionality, needs to be specified according to the model's function.
        hidden_dim       : int, optional, default: 256
            The dimensionality of the hidden layers
        num_hidden       : int, optional, default: 2
            The number of hidden layers (minimum: 1)
        activation       : string, optional, default: 'gelu'
            The activation function of the dense layers
        residual         : bool, optional, default: True
            Use residual connections in the MLP.
        spectral_norm    : bool, optional, default: False
            Use spectral normalization for the network weights, which can make
            the learned function smoother and hence more robust to perturbations.
        dropout_rate     : float, optional, default: 0.05
            Dropout rate for the hidden layers in the MLP
        zero_output_init : bool, optional, default: True
            Will initialize the last layer's kernel to zeros, which can be helpful
            when used in conjunction with coupling layers.
        """

        super().__init__(**kwargs)

        self.dim = output_dim
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
        if zero_output_init:
            output_initializer = "zeros"
        else:
            output_initializer = "glorot_uniform"
        self.output_layer = keras.layers.Dense(output_dim, kernel_initializer=output_initializer)

    def call(self, inputs, training=False):
        out = self.res_blocks(inputs, training=training)
        return self.output_layer(out)
