
import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.experimental.types import Shape
from .fixed_permutation import FixedPermutation


@register_keras_serializable(package="bayesflow.networks.coupling_flow")
class Swap(FixedPermutation):
    def build(self, input_shape: Shape) -> None:
        shift = input_shape[-1] // 2
        forward_indices = keras.ops.roll(keras.ops.arange(input_shape[-1]), shift=shift)
        inverse_indices = keras.ops.argsort(forward_indices)

        self.forward_indices = self.add_variable(
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(forward_indices),
            trainable=False
        )

        self.inverse_indices = self.add_variable(
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(inverse_indices),
            trainable=False
        )
