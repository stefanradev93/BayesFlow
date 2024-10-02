
from collections.abc import Sequence
import numpy as np
from typing import Type

from .transform import Transform



class ElementwiseTransform(Transform):
    def __init__(self, keys: str | Sequence[str] = None, exclude: str | Sequence[str] = None):
        self.keys = keys
        self.exclude = exclude

        if isinstance(self.keys, str):
            self.keys = [self.keys]

        if isinstance(self.exclude, str):
            self.exclude = [self.exclude]
        elif self.exclude is None:
            self.exclude = []

    def forward(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.keys is None:
            self.keys = list(data.keys())

        for key in self.keys:
            if key in self.exclude:
                continue

            data[key] = self._forward(key, data[key])

        return data

    def inverse(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.keys is None:
            raise ValueError("Cannot call `inverse` before calling `forward` at least once.")

        for key in self.keys:
            if key in self.exclude:
                continue

            data[key] = self._inverse(key, data[key])

        return data

    def _forward(self, key: str, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _inverse(self, key: str, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ElementwiseTransformInner:
    def forward(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ElementwiseTransform(Transform):
    def __init__(self, transform_type: type[ElementwiseTransformInner], keys: str | Sequence[str] | None = None, exclude: str | Sequence[str] | None = None):
        self.transform_type = transform_type

        self.keys = keys
        self.exclude = exclude

        if isinstance(self.keys, str):
            self.keys = [self.keys]

        if isinstance(self.exclude, str):
            self.exclude = [self.exclude]
        elif self.exclude is None:
            self.exclude = []

        if self.keys is not None:
            self.transforms = {key: self.transform_type() for key in self.keys}


        self.transforms = {}

    def forward(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
