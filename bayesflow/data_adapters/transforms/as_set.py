import numpy as np

from .elementwise_transform import ElementwiseTransform


class AsSet(ElementwiseTransform):
    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.atleast_3d(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.squeeze(data, axis=2)
