
import keras

from .coupling import Coupling
from ..permutation import Permutation
from ..transforms import Transform


class DualCoupling(keras.Layer):
    """ Implements a dual coupling layer with swap permutations. """
    def __init__(self, subnet_constructor: callable, features: int, conditions: int, transform: Transform):
        super().__init__()

        self.coupling1 = Coupling(subnet_constructor, features, conditions, transform, permutation=Permutation.swap(features))
        self.coupling2 = Coupling(subnet_constructor, features, conditions, transform, permutation=Permutation.swap(features))

    def forward(self, x, c=None):
        z, det1 = self.coupling1.forward(x, c)
        z, det2 = self.coupling2.forward(z, c)

        return z, det1 + det2

    def inverse(self, z, c=None):
        x, det2 = self.coupling2.inverse(z, c)
        x, det1 = self.coupling1.inverse(x, c)

        return x, det1 + det2
