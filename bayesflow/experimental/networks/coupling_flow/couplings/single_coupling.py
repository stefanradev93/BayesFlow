
import keras
from keras.saving import register_keras_serializable

from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import find_network, keras_kwargs
from ..invertible_layer import InvertibleLayer
from ..transforms import find_transform


@register_keras_serializable(package="bayesflow.networks.coupling_flow")
class SingleCoupling(InvertibleLayer):
    """
    Implements a single coupling layer as a composition of a subnet and a transform.

    Subnet output tensors are linearly mapped to the correct dimension.
    """
    def __init__(
        self,
        network: str = "resnet",
        transform: str = "affine",
        output_layer_kernel_init: str = "zeros",
        **kwargs
    ):
        super().__init__(**keras_kwargs(kwargs))
        self.output_projector = keras.layers.Dense(
            units=None,
            kernel_initializer=output_layer_kernel_init,
        )
        self.network = find_network(network, **kwargs.get("subnet_kwargs", {}))
        self.transform = find_transform(transform, **kwargs.get("transform_kwargs", {}))

    # noinspection PyMethodOverriding
    def build(self, x1_shape, x2_shape):
        self.output_projector.units = self.transform.params_per_dim * x2_shape[-1]

    def call(self, x1: Tensor, x2: Tensor, conditions: Tensor = None, inverse: bool = False, **kwargs) -> ((Tensor, Tensor), Tensor):
        if inverse:
            return self._inverse(x1, x2, conditions=conditions, **kwargs)
        return self._forward(x1, x2, conditions=conditions, **kwargs)

    def _forward(self, x1: Tensor, x2: Tensor, conditions: Tensor = None, **kwargs) -> ((Tensor, Tensor), Tensor):
        """ Transform (x1, x2) -> (x1, f(x2; x1)) """
        z1 = x1
        parameters = self.get_parameters(x1, conditions=conditions, **kwargs)
        z2, log_det = self.transform(x2, parameters=parameters)

        return (z1, z2), log_det

    def _inverse(self, z1: Tensor, z2: Tensor, conditions: Tensor = None, **kwargs) -> ((Tensor, Tensor), Tensor):
        """ Transform (x1, f(x2; x1)) -> (x1, x2) """
        x1 = z1
        parameters = self.get_parameters(x1, conditions=conditions, **kwargs)
        x2, log_det = self.transform(z2, parameters=parameters, inverse=True)

        return (x1, x2), log_det

    def get_parameters(self, x: Tensor, conditions: Tensor = None, **kwargs) -> dict[str, Tensor]:
        if conditions is not None:
            x = keras.ops.concatenate([x, conditions], axis=-1)

        parameters = self.output_projector(self.network(x, **kwargs))
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)

        return parameters
