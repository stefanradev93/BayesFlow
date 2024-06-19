
import keras
import numpy as np

from .simulator import Simulator
from ..types import Shape, Tensor


class TwoMoonsSimulator(Simulator):
    """ TODO: Docs """
    def sample(self, batch_shape: Shape) -> dict[str, Tensor]:
        r = keras.random.normal(batch_shape + (1,), 0.1, 0.01)
        alpha = keras.random.uniform(batch_shape + (1,), -0.5 * np.pi, 0.5 * np.pi)

        theta = keras.random.uniform(batch_shape + (2,), -1.0, 1.0)

        x1 = -keras.ops.abs(theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * keras.ops.cos(alpha) + 0.25
        x2 = (-theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * keras.ops.sin(alpha)

        x = keras.ops.concatenate([x1, x2], axis=-1)

        return dict(r=r, alpha=alpha, theta=theta, x=x)
