from collections.abc import Sequence

from .transform import Transform


class Drop(Transform):
    def __init__(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        self.keys = keys

    def forward(self, data: dict[str, any]) -> dict[str, any]:
        return {key: value for key, value in data.items() if key not in self.keys}

    def inverse(self, data: dict[str, any]) -> dict[str, any]:
        # non-invertible transform
        return data
