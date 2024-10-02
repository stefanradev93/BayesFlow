from collections.abc import Sequence
import numpy as np

from .transform import Transform


class Concatenate(Transform):
    def __init__(self, keys: Sequence[str], into: str, axis: int = -1):
        self.keys = keys
        self.into = into
        self.axis = axis

        self.splits = None

    def forward(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.splits is None:
            self.splits = [data[key].shape[self.axis] for key in self.keys]

        data[self.into] = np.concatenate([data.pop(key) for key in self.keys], axis=self.axis)

        return data

    def inverse(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.splits is None:
            raise ValueError("Cannot call `inverse` before calling `forward` at least once.")

        values = np.split(data.pop(self.into), self.splits, axis=self.axis)

        for key, value in zip(self.keys, values):
            data[key] = value

        return data
