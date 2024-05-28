
import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Tensor
from ..invertible_layer import InvertibleLayer
from ..subnets import find_subnet
from ..transforms import find_transform, Transform


@register_keras_serializable(package="bayesflow.networks.coupling_flow")
class SingleCoupling(InvertibleLayer):
    def __init__(self, subnet: keras.Layer, transform: Transform, **kwargs):
        super().__init__(**kwargs)
        self.subnet = subnet
        self.transform = transform

    @classmethod
    def new(cls, subnet: str = "resnet", transform: str = "affine", **kwargs) -> "SingleCoupling":
        transform = find_transform(transform)
        subnet = find_subnet(subnet)

        return cls(subnet, transform, **kwargs)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "SingleCoupling":
        subnet = deserialize_keras_object(config.pop("subnet"))
        transform = deserialize_keras_object(config.pop("transform"))

        return cls(subnet, transform, **config)

    def get_config(self) -> dict:
        base_config = super().get_config()

        config = {
            "subnet": serialize_keras_object(self.subnet),
            "transform": serialize_keras_object(self.transform),
        }

        return base_config | config

    def build(self, input_shape):
        # TODO: this is not ideal...
        if not hasattr(self.subnet, "build_output"):
            return

        x1_shape, x2_shape = input_shape
        x2_shape = list(x2_shape)
        x2_shape[-1] = self.transform.params_per_dim * x2_shape[-1]
        self.subnet.build_output(x2_shape)

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

        parameters = self.subnet(x)
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)

        return parameters
