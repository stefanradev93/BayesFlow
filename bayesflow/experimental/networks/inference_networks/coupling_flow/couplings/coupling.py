
import keras

from ..permutation import FixedPermutation, LearnablePermutation
from ..actnorm import ActNorm
from ..transforms import Transform


class Coupling(keras.Layer):
    """ Implements a single coupling layer, followed by a permutation. """
    def __init__(
        self,
        subnet_constructor: callable,
        target_dim: int,
        transform: Transform,
        **kwargs
    ):
        super().__init__()
        self.subnet = subnet_constructor(target_dim, **kwargs.pop('subnet_kwargs', {}))
        self.transform = transform

    def forward(self, x, c=None, **kwargs):

        x1, x2 = keras.ops.split(x, 2, axis=-1)
        z1 = x1
        parameters = self.get_parameters(x1, c, **kwargs)
        z2, log_det = self.transform.forward(x2, parameters)
        z = keras.ops.concatenate([z1, z2], axis=-1)
        return z, log_det

    def inverse(self, z, c=None):

        z1, z2 = keras.ops.split(z, 2, axis=-1)
        x1 = z1
        parameters = self.get_parameters(x1, c)
        x2, log_det = self.transform.inverse(z2, parameters)
        x = keras.ops.concatenate([x1, x2], axis=-1)
        return x, log_det

    def get_parameters(self, x, c=None, **kwargs):
        if c is not None:
            x = keras.ops.concatenate([x, c], axis=-1)

        parameters = self.subnet(x, **kwargs)
        parameters = self.transform.constrain_parameters(parameters)

        return parameters
