
import keras


class ConfigurableHiddenBlock(keras.layers.Layer):
    def __init__(
        self,
        num_units,
        activation="relu",
        residual=True,
        dropout_rate=0.05,
        spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.activation_fn = keras.activations.get(activation)
        self.residual = residual
        self.spectral_norm = spectral_norm
        self.dense_with_dropout = keras.Sequential()

        if spectral_norm:
            self.dense_with_dropout.add(keras.layers.SpectralNormalization(keras.layers.Dense(num_units)))
        else:
            self.dense_with_dropout.add(keras.layers.Dense(num_units))
        self.dense_with_dropout.add(keras.layers.Dropout(dropout_rate))

    def call(self, inputs, training=False):
        x = self.dense_with_dropout(inputs, training=training)
        if self.residual:
            x = x + inputs
        return self.activation_fn(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "residual": self.residual,
            "spectral_norm": self.spectral_norm,
            "activation_fn": keras.saving.serialize_keras_object(self.activation_fn),
            "dense_with_dropout": keras.saving.serialize_keras_object(self.dense_with_dropout)
        })
        return config
