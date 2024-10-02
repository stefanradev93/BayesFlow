
from collections.abc import Callable, Sequence
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .transforms import (
    Concatenate,
    Transform,
)


@serializable(package="bayesflow.data_adapters")
class DataAdapter:
    def __init__(self, transforms: Sequence[Transform] | None = None):
        self.transforms = transforms or []

    @classmethod
    def default(cls):
        instance = cls()
        instance.expand_scalars()
        instance.convert_dtypes(from_dtype="float64", to_dtype="float32")
        instance.standardize()

        return instance

    def forward(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        data = data.copy()

        for transform in self.transforms:
            data = transform(data)

        return data

    def inverse(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        data = data.copy()

        for transform in reversed(self.transforms):
            data = transform(data, inverse=True)

        return data

    def __call__(self, data: dict[str, np.ndarray], inverse: bool = False) -> dict[str, np.ndarray]:
        if inverse:
            return self.inverse(data)

        return self.forward(data)

    def add_transform(self, transform: Transform):
        self.transforms.append(transform)
        return self

    def apply(self, forward: Callable, inverse: Callable):
        self.transforms.append(LambdaTransform(forward, inverse))
        return self

    def clear(self):
        self.transforms = []
        return self

    def concatenate(self, keys: Sequence[str], into: str, axis: int = -1):
        self.transforms.append(Concatenate(keys, into, axis))
        return self

    def convert_dtypes(self, keys: str | Sequence[str] = None, exclude: str | Sequence[str] = None, *, from_dtype: str, to_dtype: str):
        ...
        return self

    def constrain(self, keys: Sequence[str], *, lower: float | np.ndarray = None, upper: float | np.ndarray = None, method: str):
        ...
        return self

    def expand_scalars(self, keys: str | Sequence[str] = None, exclude: str | Sequence[str] = None):
        self.transforms.append(...)  # convert to numpy arrays
        self.transforms.append(Reshape(...))  # reshape (batch_size,) to (batch_size, 1)
        return self

    def rename(self, from_key: str, to_key: str):
        self.transforms.append(Concatenate([from_key], into=to_key))
        return self

    def standardize(self, keys: str | Sequence[str] = None, exclude: str | Sequence[str] = None, means: Mapping[str, np.ndarray] = None, stds: Mapping[str, np.ndarray] = None):
        self.transforms.append(Standardize(keys, exclude, ))
        return self



# data_adapter = bf.ContinuousApproximator.build_data_adapter(
#     inference_variables=["theta", "alpha"],
# )

approximator = bf.ContinuousApproximator()
approximator.build_data_adapter(
    transforms=[bf.data_adapters.transforms.Standardize(["theta", "x"])],
)



# example usage
data_adapter = DataAdapter()
data_adapter.expand_scalars()  # (batch_size,) -> (batch_size, 1)
data_adapter.convert_dtypes(from_dtype="float64", to_dtype="float32")
data_adapter.standardize(exclude="num_obs")
data_adapter.concatenate(["theta", "r", "alpha"], into="inference_variables")
data_adapter.rename("x", "inference_conditions")


# or use the builder pattern
data_adapter = DataAdapter.default() \
    .concatenate(["theta", "r", "alpha"], into="inference_variables") \
    .rename("x", "inference_conditions")




from typing import Generic, TypeVar


TRaw = TypeVar("TRaw")
TProcessed = TypeVar("TProcessed")


class DataAdapter(Generic[TRaw, TProcessed]):
    """Construct and deconstruct deep-learning ready data from and into raw data."""

    def configure(self, raw_data: TRaw) -> TProcessed:
        """Construct deep-learning ready data from raw data."""
        raise NotImplementedError

    def deconfigure(self, processed_data: TProcessed) -> TRaw:
        """Reconstruct raw data from deep-learning ready processed data.
        Note that configuration is not required to be bijective, so this method is only meant to be a 'best effort'
        attempt, and may return incomplete or different raw data.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "DataAdapter":
        """Construct a data adapter from a configuration dictionary."""
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return a configuration dictionary."""
        raise NotImplementedError
