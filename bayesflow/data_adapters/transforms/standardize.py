# from collections.abc import Mapping, Sequence
# from keras.saving import (
#     deserialize_keras_object as deserialize,
#     register_keras_serializable as serializable,
#     serialize_keras_object as serialize,
# )
# import numpy as np
#
# from .transform import ElementwiseTransform
#
#
# @serializable(package="bayesflow.data_adapters")
# class Standardize(ElementwiseTransform):
#     """Normalizes a parameter to have zero mean and unit standard deviation.
#     By default, this is lazily initialized; the mean and standard deviation are computed from the first batch of data.
#     For eager initialization, pass the mean and standard deviation to the constructor.
#     """
#
#     def __init__(
#         self,
#         parameters: str | Sequence[str] | None = None,
#         /,
#         *,
#         means: Mapping[str, np.ndarray] = None,
#         stds: Mapping[str, np.ndarray] = None,
#     ):
#         super().__init__(parameters)
#         self.means = means or {}
#         self.stds = stds or {}
#
#     @classmethod
#     def from_config(cls, config: dict, custom_objects=None) -> "Standardize":
#         return cls(
#             deserialize(config["parameters"], custom_objects),
#             means=deserialize(config["means"], custom_objects),
#             stds=deserialize(config["stds"], custom_objects),
#         )
#
#     def forward(self, parameter_name: str, parameter_value: np.ndarray) -> np.ndarray:
#         if parameter_name not in self.means:
#             self.means[parameter_name] = np.mean(
#                 parameter_value, axis=tuple(range(parameter_value.ndim)), keepdims=True
#             )
#         if parameter_name not in self.stds:
#             self.stds[parameter_name] = np.std(parameter_value, axis=tuple(range(parameter_value.ndim)), keepdims=True)
#
#         return (parameter_value - self.means[parameter_name]) / self.stds[parameter_name]
#
#     def get_config(self) -> dict:
#         return {
#             "parameters": serialize(self.parameters),
#             "means": serialize(self.means),
#             "stds": serialize(self.stds),
#         }
#
#     def inverse(self, parameter_name: str, parameter_value: np.ndarray) -> np.ndarray:
#         if not self.means or not self.stds:
#             raise ValueError(
#                 f"Cannot call `inverse` before calling `forward` at least once for parameter {parameter_name}."
#             )
#
#         return parameter_value * self.stds[parameter_name] + self.means[parameter_name]

from collections.abc import Mapping, Sequence
import numpy as np

from .elementwise_transform import ElementwiseTransformInner


class Standardize(ElementwiseTransformInner):
    def __init__(self, mean: int | float | np.ndarray = None, std: int | float | np.ndarray = None, axis: int = 0):
        self.mean = mean
        self.std = std
        self.axis = axis

    def forward(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            self.mean = np.mean(data, axis=self.axis, keepdims=True)

        if self.std is None:
            self.std = np.std(data, axis=self.axis, keepdims=True)

        return (data - self.mean) / self.std

    def inverse(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Cannot call `inverse` before calling `forward` at least once.")

        return data * self.std + self.mean



# class Standardize(ElementwiseTransform):
#     def __init__(self, keys: str | Sequence[str] = None, exclude: str | Sequence[str] = None, means: Mapping[str, np.ndarray] = None, stds: Mapping[str, np.ndarray] = None, axis: int | tuple[int, ...] = 0):
#         super().__init__(keys, exclude)
#
#         if bool(means) ^ bool(stds):
#             raise ValueError("Either both or neither of `means` and `stds` must be provided.")
#
#         elif means:
#             if set(means) != set(stds):
#                 raise ValueError(f"The keys of `means` and `stds` must identical.")
#
#         self.means = means or {}
#         self.stds = stds or {}
#
#         self.axis = axis
#
#     def _forward(self, key: str, value: np.ndarray) -> np.ndarray:
#         if key not in self.means:
#             self.means[key] = np.mean(value, axis=self.axis, keepdims=True)
#
#         if key not in self.stds:
#             self.stds[key] = np.std(value, axis=self.axis, keepdims=True)
#
#         return (value - self.means[key]) / self.stds[key]
#
#     def _inverse(self, key: str, value: np.ndarray) -> np.ndarray:
#         if not self.means or not self.stds:
#             raise ValueError(f"Cannot call `inverse` before calling `forward` at least once for parameter {key}.")
#
#         return value * self.stds[key] + self.means[key]
#
#
# class Standardize(Transform):
#     def __init__(self, keys: str | Sequence[str] = None, exclude: str | Sequence[str] = None, means: Mapping[str, np.ndarray] = None, stds: Mapping[str, np.ndarray] = None, axis: int = 0):
#         self.keys = keys
#         self.exclude = exclude
#
#         if isinstance(self.keys, str):
#             self.keys = [self.keys]
#
#         if isinstance(self.exclude, str):
#             self.exclude = [self.exclude]
#         elif self.exclude is None:
#             self.exclude = []
#
#         if bool(means) ^ bool(stds):
#             raise ValueError("Either both or neither of `means` and `stds` must be provided.")
#         elif means:
#             if set(means) != set(stds):
#                 raise ValueError(f"The keys of `means` and `stds` must be the same.")
#
#         self.means = means
#         self.stds = stds
#
#         self.axis = axis
#
#     def forward(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
#         if self.keys is None:
#             self.keys = list(data.keys())
#
#         if self.means is None:
#             self.means = {key: np.mean(data[key], axis=self.axis, keepdims=True) for key in self.keys}
#
#         if self.stds is None:
#             self.stds = {key: np.std(data[key], axis=self.axis, keepdims=True) for key in self.keys}
#
#         for key in self.keys:
#             if key in self.exclude:
#                 continue
#
#             if key not in self.means:
#
#
#         return data



