from .transform import Transform


class Rename(Transform):
    def __init__(self, from_key: str, to_key: str):
        super().__init__()
        self.from_key = from_key
        self.to_key = to_key

    def forward(self, data: dict[str, any]) -> dict[str, any]:
        data = data.copy()
        data[self.to_key] = data.pop(self.from_key)
        return data

    def inverse(self, data: dict[str, any]) -> dict[str, any]:
        data = data.copy()
        data[self.from_key] = data.pop(self.to_key)
        return data
