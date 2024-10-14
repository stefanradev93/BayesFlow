from keras.saving import register_keras_serializable as serializable
import numpy as np


@serializable(package="bayesflow.data_adapters")
class ElementwiseTransform:
    def __call__(self, data: np.ndarray, inverse: bool = False):
        if inverse:
            return self.inverse(data)

        return self.forward(data)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ElementwiseTransform":
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

    def forward(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
