
import keras

from bayesflow.experimental.types import Tensor


class Transform(keras.layers.Layer):
    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError

    def inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError
