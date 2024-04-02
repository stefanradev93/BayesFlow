
import keras

from bayesflow.experimental.types import Tensor


class Transform(keras.layers.Layer):
    def forward(self, x: Tensor, parameters: dict[str, Tensor]) -> Tensor:
        z, logdet = self.forward_jacobian(x, parameters)
        return z

    def inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> Tensor:
        x, logdet = self.inverse_jacobian(z, parameters)
        return x

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def forward_jacobian(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError

    def inverse_jacobian(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError
