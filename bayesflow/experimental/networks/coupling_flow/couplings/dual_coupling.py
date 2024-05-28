
import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Tensor
from .single_coupling import SingleCoupling
from ..invertible_layer import InvertibleLayer


@register_keras_serializable(package="bayesflow.networks.coupling_flow")
class DualCoupling(InvertibleLayer):
    def __init__(self, coupling1: SingleCoupling, coupling2: SingleCoupling, pivot: int = None, **kwargs):
        super().__init__(**kwargs)
        self.coupling1 = coupling1
        self.coupling2 = coupling2
        self.pivot = pivot

    @classmethod
    def new(cls, *args, **kwargs) -> "DualCoupling":
        """ Construct a new DualCoupling from hyperparameters. """
        coupling1 = SingleCoupling.new(*args, **kwargs)
        coupling2 = SingleCoupling.new(*args, **kwargs)

        return cls(coupling1, coupling2, **kwargs)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "DualCoupling":
        coupling1 = deserialize_keras_object(config.pop("coupling1"))
        coupling2 = deserialize_keras_object(config.pop("coupling2"))
        pivot = config.pop("pivot")

        return cls(coupling1, coupling2, pivot=pivot, **config)

    def get_config(self) -> dict:
        base_config = super().get_config()

        config = {
            "coupling1": serialize_keras_object(self.coupling1),
            "coupling2": serialize_keras_object(self.coupling2),
            "pivot": self.pivot,
        }

        return base_config | config

    def build(self, input_shape):
        self.pivot = input_shape[-1] // 2

        x1_shape = list(input_shape)
        x2_shape = list(input_shape)

        x1_shape[-1] = self.pivot
        x2_shape[-1] = input_shape[-1] - self.pivot

        self.coupling1.build((x1_shape, x2_shape))
        self.coupling2.build((x2_shape, x1_shape))

        self.built = True

    def call(self, xz: Tensor, conditions: any = None, inverse: bool = False) -> (Tensor, Tensor):
        if inverse:
            return self._inverse(xz, conditions=conditions)
        return self._forward(xz, conditions=conditions)

    def _forward(self, x: Tensor, conditions: any = None) -> (Tensor, Tensor):
        """ Transform (x1, x2) -> (g(x1; f(x2; x1)), f(x2; x1)) """
        x1, x2 = x[..., :self.pivot], x[..., self.pivot:]
        (z1, z2), log_det1 = self.coupling1(x1, x2, conditions=conditions)
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions)

        z = keras.ops.concatenate([z1, z2], axis=-1)
        log_det = log_det1 + log_det2

        return z, log_det

    def _inverse(self, z: Tensor, conditions: any = None) -> (Tensor, Tensor):
        """ Transform (g(x1; f(x2; x1)), f(x2; x1)) -> (x1, x2) """
        z1, z2 = z[..., :self.pivot], z[..., self.pivot:]
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, inverse=True)
        (x1, x2), log_det1 = self.coupling1(z1, z2, conditions=conditions, inverse=True)

        x = keras.ops.concatenate([x1, x2], axis=-1)
        log_det = log_det1 + log_det2

        return x, log_det
