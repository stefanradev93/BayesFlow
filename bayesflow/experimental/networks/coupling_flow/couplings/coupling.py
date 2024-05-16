
import keras
from keras import ops

from ..transforms import Transform, find_transform
from ..subnets import find_subnet


class Coupling(keras.Layer):
    """ Implements a single coupling layer that transforms half of its input through a coupling transform."""
    def __init__(
        self,
        subnet_builder: str,
        half_dim: int,
        transform: str,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.half_dim = half_dim
        self.subnet = find_subnet(
            subnet=subnet_builder,
            transform=transform,
            output_dim=half_dim,
            **kwargs.pop('subnet_kwargs', {})
        )
        self.transform = find_transform(transform)

    def call(self, x, c=None, forward=True, **kwargs):
        if forward:
            return self.forward(x, c, **kwargs)
        return self.inverse(x, c)

    def forward(self, x, c=None, **kwargs):

        x1, x2 = x[..., :self.half_dim], x[..., self.half_dim:]
        z2 = x2
        parameters = self.get_parameters(x2, c, **kwargs)
        z1, log_det = self.transform.forward(x1, parameters)
        z = ops.concatenate([z1, z2], axis=-1)
        return z, log_det

    def inverse(self, z, c=None):
        z1, z2 = z[..., :self.half_dim], z[..., self.half_dim:]
        x2 = z2
        parameters = self.get_parameters(x2, c)
        x1, log_det = self.transform.inverse(z1, parameters)
        x = ops.concatenate([x1, x2], axis=-1)
        return x, log_det

    def get_parameters(self, x, c=None, **kwargs):
        if c is not None:
            x = ops.concatenate([x, c], axis=-1)

        parameters = self.subnet(x, **kwargs)
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)

        return parameters
