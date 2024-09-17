import numpy as np

from .simulator import Simulator
from ..types import Shape


class TwoMoonsSimulator(Simulator):
    """TODO: Docs"""

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        r = np.random.normal(0.1, 0.01, size=batch_shape + (1,))
        alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, size=batch_shape + (1,))
        theta = np.random.uniform(-1.0, 1.0, size=batch_shape + (2,))

        x1 = -np.abs(theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * np.cos(alpha) + 0.25
        x2 = (-theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * np.sin(alpha)

        x = np.concatenate([x1, x2], axis=-1)

        r = r.astype("float32")
        alpha = alpha.astype("float32")
        theta = theta.astype("float32")
        x = x.astype("float32")

        return dict(r=r, alpha=alpha, theta=theta, x=x)
