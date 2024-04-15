
import keras

from ..permutation import Permutation
from ..transforms import Transform


class Coupling(keras.Layer):
    """ Implements a single coupling layer, followed by a permutation. """
    def __init__(self, subnet_constructor: callable, features: int, conditions: int, transform: Transform, permutation: Permutation):
        super().__init__()
        self.subnet = subnet_constructor(features // 2 + conditions, features - features // 2)
        self.transform = transform
        self.permutation = permutation

    def forward(self, x, c=None):
        x1, x2 = keras.ops.split(x, 2, axis=1)

        z1 = x1
        parameters = self.get_parameters(x1, c)
        z2, logdet = self.transform(x2, **parameters)

        z = keras.ops.concatenate([z1, z2], dim=1)
        z = self.permutation(z)

        return z, logdet

    def inverse(self, z, c=None):
        z = self.permutation.inverse(z)

        z1, z2 = keras.ops.split(z, 2, axis=1)

        x1 = z1
        parameters = self.get_parameters(x1, c)
        x2, logdet = self.transform.inverse(z2, **parameters)

        x = keras.ops.concatenate([x1, x2], dim=1)

        return x, logdet

    def get_parameters(self, x, c=None):
        if c is not None:
            x = keras.ops.concatenate([x, c], axis=1)

        parameters = self.subnet(x)
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)

        return parameters
