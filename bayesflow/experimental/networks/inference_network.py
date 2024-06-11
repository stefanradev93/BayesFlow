
from typing import Tuple, Union

import keras

from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import find_distribution


class InferenceNetwork(keras.Layer):
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

    def sample(self, num_samples: int, conditions: Tensor = None, **kwargs) -> Tensor:
        samples = self.base_distribution.sample((num_samples,))
        samples = self(samples, conditions=conditions, inverse=True, jacobian=False, **kwargs)
        return samples

    def log_prob(self, samples: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        samples, log_det = self(samples, conditions=conditions, inverse=False, jacobian=True, **kwargs)
        log_prob = self.base_distribution.log_prob(samples)
        return log_prob + log_det

    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        raise NotImplementedError
