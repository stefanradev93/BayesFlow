
import keras

from bayesflow.experimental.types import Tensor


class InferenceNetwork(keras.Model):
    def __init__(self, base_distribution, **kwargs):
        super().__init__(**kwargs)
        self.base_distribution = base_distribution

    def call(self, xz: Tensor, inverse: bool = False, **kwargs) -> Tensor | (Tensor, Tensor):
        if inverse:
            return self._inverse(xz, **kwargs)
        return self._forward(xz, **kwargs)

    def _forward(self, x: Tensor, **kwargs) -> Tensor | (Tensor, Tensor):
        raise NotImplementedError

    def _inverse(self, z: Tensor, **kwargs) -> Tensor | (Tensor, Tensor):
        raise NotImplementedError

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        samples = self.base_distribution.sample((num_samples,))
        return self(samples, inverse=True, jacobian=False, **kwargs)

    def log_prob(self, x: Tensor, **kwargs) -> Tensor:
        samples, log_det = self(x, inverse=False, jacobian=True, **kwargs)
        log_prob = self.base_distribution.log_prob(samples)
        return log_prob + log_det
