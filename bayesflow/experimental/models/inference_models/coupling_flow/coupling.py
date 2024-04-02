
import keras

from .permutations import Permutation
from .transforms import Transform


class Coupling(keras.layers.Layer):
    def __init__(self, transform: Transform, network: keras.layers.Layer, permutation: Permutation):
        super().__init__()
        self.transform = transform
        self.network = network
        self.permutation = permutation

    def forward(self, x):
        x1, x2 = keras.ops.split(x, 2, axis=1)

        z1 = x1
        parameters = self.network(x1)
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)
        z2 = self.transform.forward(x2, parameters)

        z = keras.ops.concatenate([z1, z2], axis=1)

        z = self.permutation.forward(z)

        return z

    def inverse(self, z):
        z = self.permutation.inverse(z)

        z1, z2 = keras.ops.split(z, 2, axis=1)

        x1 = z1
        parameters = self.network(x1)
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)
        x2 = self.transform.inverse(z2, parameters)

        x = keras.ops.concatenate([x1, x2], axis=1)

        return x

    def forward_jacobian(self, x):
        x1, x2 = keras.ops.split(x, 2, axis=1)

        z1 = x1
        parameters = self.network(x1)
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)
        z2, logdet = self.transform.forward_jacobian(x2, parameters)

        z = keras.ops.concatenate([z1, z2], axis=1)

        z = self.permutation.forward(z)

        return z, logdet

    def inverse_jacobian(self, z):
        z = self.permutation.inverse(z)

        z1, z2 = keras.ops.split(z, 2, axis=1)

        x1 = z1
        parameters = self.network(x1)
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)
        x2, logdet = self.transform.inverse_jacobian(z2, parameters)

        x = keras.ops.concatenate([x1, x2], axis=1)

        return x, logdet
