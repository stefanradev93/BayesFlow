
from typing import Tuple, Union

import keras
from keras.saving import (
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Tensor


@register_keras_serializable(package="bayesflow.networks")
class InferenceNetwork(keras.Model):
    def __init__(self, base_distribution: str = "normal", **kwargs):
        super().__init__(**kwargs)
        # TODO: get the actual distribution object from the string representation
        self.base_distribution = base_distribution

    def get_config(self) -> dict:
        base_config = super().get_config()
        # TODO: get the string representation of the distribution object
        config = {"base_distribution": serialize_keras_object(self.base_distribution)}
        return base_config | config

    def call(self, xz: Tensor, inverse: bool = False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if inverse:
            return self._inverse(xz, **kwargs)
        return self._forward(xz, **kwargs)

    def _forward(self, x: Tensor, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def _inverse(self, z: Tensor, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        samples = self.base_distribution.sample((num_samples,))
        return self(samples, inverse=True, jacobian=False, **kwargs)

    def log_prob(self, x: Tensor, **kwargs) -> Tensor:
        samples, log_det = self(x, inverse=False, jacobian=True, **kwargs)
        log_prob = self.base_distribution.log_prob(samples)
        return log_prob + log_det
