
import keras
from keras import layers, regularizers
from keras.saving import (
    register_keras_serializable,
    serialize_keras_object
)

from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import find_pooling


@register_keras_serializable(package="bayesflow.networks.deep_set")
class InvariantModule(keras.Layer):
    """Implements an invariant module performing a permutation-invariant transform.

    For details and rationale, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(
        self,
        num_dense_inner: int = 2,
        num_dense_outer: int = 2,
        units_inner: int = 128,
        units_outer: int = 128,
        activation: str | callable = "gelu",
        kernel_regularizer: regularizers.Regularizer | None = None,
        kernel_initializer: str = "he_uniform",
        bias_regularizer: regularizers.Regularizer | None = None,
        dropout: float = 0.05,
        pooling: str = "mean",
        spectral_normalization: bool = False,
        **kwargs
    ):
        """Creates an invariant module according to [1] which represents a learnable permutation-invariant
        function with an option for learnable pooling.

        Parameters
        ----------
        # TODO
        """
        super().__init__(**kwargs)

        # Inner fully connected net for sum decomposition: inner( pooling( inner(set) ) )
        self.inner_fc = keras.Sequential(name="InvariantInnerFC")
        for _ in range(num_dense_inner):
            layer = layers.Dense(
                units=units_inner,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                bias_regularizer=bias_regularizer
            )
            if spectral_normalization:
                layer = layers.SpectralNormalization(layer)
            self.inner_fc.add(layer)

        # Outer fully connected net for sum decomposition: inner( pooling( inner(set) ) )
        self.outer_fc = keras.Sequential(name="InvariantOuterFC")
        for _ in range(num_dense_outer):

            self.outer_fc.add(layers.Dropout(dropout))

            layer = layers.Dense(
                units=units_outer,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                bias_regularizer=bias_regularizer
            )
            if spectral_normalization:
                layer = layers.SpectralNormalization(layer)
            self.outer_fc.add(layer)

        # Pooling function as keras layer for sum decomposition: inner( pooling( inner(set) ) )
        self.pooling_layer = find_pooling(pooling, **kwargs)

    def call(self, input_set: Tensor, **kwargs) -> Tensor:
        """Performs the forward pass of a learnable invariant transform.

        Parameters
        ----------
        input_set : tf.Tensor
            Input of shape (batch_size,..., input_dim)

        Returns
        -------
        set_summary : tf.Tensor
            Output of shape (batch_size,..., out_dim)
        """

        set_summary = self.inner_fc(input_set, training=kwargs.get("training", False))
        set_summary = self.pooling_layer(set_summary, training=kwargs.get("training", False))
        set_summary = self.outer_fc(set_summary, training=kwargs.get("training", False))
        return set_summary

    def build(self, input_shape):
        super().build(input_shape)
        self(keras.KerasTensor(input_shape))

    def get_config(self):
        config = super().get_config()
        config.update({
            "inner_fc": serialize_keras_object(self.inner_fc),
            "outer_fc": serialize_keras_object(self.outer_fc),
            "pooling_layer": serialize_keras_object(self.pooling_layer)
        })
        return config
