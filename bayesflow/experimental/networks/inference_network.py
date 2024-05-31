
from typing import Tuple, Union

import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import find_distribution


@register_keras_serializable(package="bayesflow.networks")
class InferenceNetwork(keras.Model):
    def __init__(self, base_distribution: str = "normal", **kwargs):
        super().__init__(**kwargs)
        self.base_distribution = find_distribution(base_distribution)

    def build(self, input_shape):
        super().build(input_shape)
        self.base_distribution.build(input_shape)

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

    def train_step(self, data):
        # hack to avoid the call method in super().train_step()
        call = self.call
        self.call = lambda *args, **kwargs: None
        rv = super().train_step(data)
        self.call = call

        return rv
