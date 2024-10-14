import numpy as np

from .elementwise_transform import ElementwiseTransform


class ConvertDType(ElementwiseTransform):
    def __init__(self, from_dtype: str, to_dtype: str):
        super().__init__()

        self.from_dtype = from_dtype
        self.to_dtype = to_dtype

    def forward(self, data: np.ndarray) -> np.ndarray:
        return data.astype(self.to_dtype)

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data.astype(self.from_dtype)
