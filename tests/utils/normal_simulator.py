import keras

from bayesflow.simulators import Simulator
from bayesflow.types import Shape, Tensor


class NormalSimulator(Simulator):
    """TODO: Docstring"""

    def sample(self, batch_shape: Shape, num_observations: int = 32) -> dict[str, Tensor]:
        mean = keras.random.normal(batch_shape + (2,), 0.0, 0.1)
        mean = keras.ops.repeat(mean[:, None], num_observations, 1)

        std = keras.ops.exp(keras.random.normal(batch_shape + (2,), 0.0, 0.1))
        std = keras.ops.repeat(std[:, None], num_observations, 1)

        noise = keras.random.normal(batch_shape + (num_observations, 2))

        x = mean + std * noise
        return dict(mean=mean, std=std, x=x)
