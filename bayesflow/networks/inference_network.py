import keras

from bayesflow.types import Tensor
from bayesflow.utils import find_distribution


class InferenceNetwork(keras.Layer):
    def __init__(self, base_distribution: str = "normal", **kwargs):
        super().__init__(**kwargs)
        self.base_distribution = find_distribution(base_distribution)

    def build(self, xz_shape, **kwargs):
        self.base_distribution.build(xz_shape)

    def call(self, xz: Tensor, inverse: bool = False, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        if inverse:
            return self._inverse(xz, **kwargs)
        return self._forward(xz, **kwargs)

    def _forward(self, x: Tensor, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _inverse(self, z: Tensor, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def sample(self, num_samples: int, conditions: Tensor = None, **kwargs) -> Tensor:
        samples = self.base_distribution.sample((num_samples,))
        samples = self(samples, conditions=conditions, inverse=True, density=False, **kwargs)
        return samples

    def log_prob(self, samples: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        _, log_density = self(samples, conditions=conditions, inverse=False, density=True, **kwargs)
        return log_density

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        metrics = {}

        if stage != "training" and any(self.metrics):
            # compute sample-based metrics
            samples = self.sample(keras.ops.shape(x)[0], conditions=conditions)

            for metric in self.metrics:
                metrics[metric.name] = metric(samples, x)

        return metrics
