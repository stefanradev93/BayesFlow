import numpy as np


class Transform:
    def __init__(self, parameter_name: str):
        self.parameter_name = parameter_name

    def __call__(self, data: dict[str, np.ndarray], inverse: bool = False) -> dict[str, np.ndarray]:
        """Apply the transform to the data"""
        data = data.copy()

        if self.parameter_name not in data:
            # partial data e.g. when configuring at inference time
            # if the parameter is not present, there is nothing to do
            return data

        if inverse:
            data[self.parameter_name] = self.inverse(data[self.parameter_name])
        else:
            data[self.parameter_name] = self.forward(data[self.parameter_name])

        return data

    def forward(self, parameter: np.ndarray) -> np.ndarray:
        """Implements the forward direction of the transform"""
        raise NotImplementedError

    def inverse(self, parameter: np.ndarray) -> np.ndarray:
        """Implements the inverse direction of the transform"""
        raise NotImplementedError

    def invert(self) -> None:
        """Invert the transform, in-place"""
        self.forward, self.inverse = self.inverse, self.forward
