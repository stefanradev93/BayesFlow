
import keras


class ConfigurableHiddenBlock(keras.Layer):
    def __init__(
        self,
        num_units,
        activation="gelu",
        residual=True,
        dropout_rate=0.05,
        spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.activation_fn = keras.activations.get(activation)
        self.residual = residual
        self.dense_with_dropout = keras.Sequential()
        if spectral_norm:
            self.dense_with_dropout.add(keras.layers.SpectralNormalization(keras.layers.Dense(num_units)))
        else:
            self.dense_with_dropout.add(keras.layers.Dense(num_units))
        self.dense_with_dropout.add(keras.layers.Dropout(dropout_rate))

    def call(self, inputs, **kwargs):
        x = self.dense_with_dropout(inputs, **kwargs)
        if self.residual:
            x = x + inputs
        return self.activation_fn(x)
