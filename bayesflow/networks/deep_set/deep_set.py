import keras
from keras import layers
from keras.saving import register_keras_serializable, serialize_keras_object

from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs
from .invariant_module import InvariantModule
from .equivariant_module import EquivariantModule


@register_keras_serializable(package="bayesflow.networks")
class DeepSet(keras.Model):
    def __init__(
        self,
        summary_dim: int = 16,
        depth: int = 2,
        inner_pooling: str | keras.Layer = "mean",
        output_pooling: str | keras.Layer = "mean",
        num_dense_equivariant: int = 2,
        num_dense_invariant_inner: int = 2,
        num_dense_invariant_outer: int = 2,
        units_equivariant: int = 128,
        units_invariant_inner: int = 128,
        units_invariant_outer: int = 128,
        activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        dropout: float = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """
        #TODO
        """

        super().__init__(**keras_kwargs(kwargs))

        # Stack of equivariant modules for a many-to-many learnable transformation
        self.equivariant_modules = keras.Sequential()
        for i in range(depth):
            equivariant_module = EquivariantModule(
                num_dense_equivariant=num_dense_equivariant,
                num_dense_invariant_inner=num_dense_invariant_inner,
                num_dense_invariant_outer=num_dense_invariant_outer,
                units_equivariant=units_equivariant,
                units_invariant_inner=units_invariant_inner,
                units_invariant_outer=units_invariant_outer,
                activation=activation,
                kernel_initializer=kernel_initializer,
                spectral_normalization=spectral_normalization,
                dropout=dropout,
                pooling=inner_pooling,
                **kwargs,
            )
            self.equivariant_modules.add(equivariant_module)

        # Invariant module for a many-to-one transformation
        self.invariant_module = InvariantModule(
            num_dense_inner=num_dense_invariant_inner,
            num_dense_outer=num_dense_invariant_outer,
            units_inner=units_invariant_inner,
            units_outer=units_invariant_outer,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            pooling=output_pooling,
            spectral_normalization=spectral_normalization,
            **kwargs,
        )

        # Output linear layer to project set representation down to "summary_dim" learned summary statistics
        self.output_projector = layers.Dense(summary_dim, activation="linear")
        self.summary_dim = summary_dim

    def call(self, input_set: Tensor, **kwargs) -> Tensor:
        """Performs the forward pass of a learnable deep invariant transformation consisting of
        a sequence of equivariant transforms followed by an invariant transform.

        Parameters
        ----------
        input_set : KerasTensor
            Input set of shape (batch_size, ..., set_size, obs_dim)

        Returns
        -------
        out : KerasTensor
            Output representation of shape (batch_size, ..., set_size, obs_dim)
        """

        transformed_set = self.equivariant_modules(input_set, **kwargs)
        set_representation = self.invariant_module(transformed_set, **kwargs)
        set_representation = self.output_projector(set_representation)

        return set_representation

    def build(self, input_shape):
        self.call(keras.ops.zeros(input_shape))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "invariant_module": serialize_keras_object(self.equivariant_modules),
                "equivariant_fc": serialize_keras_object(self.invariant_module),
                "output_projector": serialize_keras_object(self.output_projector),
                "summary_dim": serialize_keras_object(self.summary_dim),
            }
        )
        return config
