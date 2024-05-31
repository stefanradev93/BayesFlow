
import keras
from keras.saving import (
    register_keras_serializable
)

from bayesflow.experimental.types import Tensor
from ..invertible_layer import InvertibleLayer
from ..subnets import find_subnet
from ..transforms import find_transform


@register_keras_serializable(package="bayesflow.networks.coupling_flow")
class SingleCoupling(InvertibleLayer):
    """
    Implements a single coupling layer as a composition of a subnet and a transform.

    Subnet output tensors are linearly mapped to the correct dimension.
    """
    def __init__(self, subnet: str = "resnet", transform: str = "affine", **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(None)
        self.subnet = find_subnet(subnet)
        self.transform = find_transform(transform)

    # noinspection PyMethodOverriding
    def build(self, x1_shape, x2_shape):
        self.dense.units = self.transform.params_per_dim * x2_shape[-1]

    def call(self, x1: Tensor, x2: Tensor, conditions: any = None, inverse: bool = False) -> ((Tensor, Tensor), Tensor):
        if inverse:
            return self._inverse(x1, x2, conditions=conditions)
        return self._forward(x1, x2, conditions=conditions)

    def _forward(self, x1: Tensor, x2: Tensor, conditions: any = None) -> ((Tensor, Tensor), Tensor):
        """ Transform (x1, x2) -> (x1, f(x2; x1)) """
        z1 = x1
        parameters = self.get_parameters(x1, conditions)
        z2, log_det = self.transform(x2, parameters=parameters)

        return (z1, z2), log_det

    def _inverse(self, z1: Tensor, z2: Tensor, conditions: any = None) -> ((Tensor, Tensor), Tensor):
        """ Transform (x1, f(x2; x1)) -> (x1, x2) """
        x1 = z1
        parameters = self.get_parameters(x1, conditions)
        x2, log_det = self.transform(z2, parameters=parameters, inverse=True)

        return (x1, x2), log_det

    def get_parameters(self, x, conditions: any = None) -> dict[str, Tensor]:
        # TODO: pass conditions to subnet via kwarg if possible
        if keras.ops.is_tensor(conditions):
            x = keras.ops.concatenate([x, conditions], axis=-1)

        parameters = self.dense(self.subnet(x))
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)

        return parameters
