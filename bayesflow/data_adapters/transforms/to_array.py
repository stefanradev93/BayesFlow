from numbers import Number

import numpy as np

from .elementwise_transform import ElementwiseTransform


class ToArray(ElementwiseTransform):
    def __init__(self):
        super().__init__()
        self.original_type = None

    def forward(self, data: any) -> np.ndarray:
        if self.original_type is None:
            self.original_type = type(data)

        return np.asarray(data)

    def inverse(self, data: np.ndarray) -> any:
        if self.original_type is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        if issubclass(self.original_type, Number):
            return self.original_type(data.item())

        # cannot invert
        return data
