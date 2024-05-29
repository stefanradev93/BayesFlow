
from typing import Tuple, Union

import keras
from keras.saving import (
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Tensor


class SummaryNetwork(keras.Model):
    def call(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def compute_loss(self, **kwargs) -> Tensor:
        raise NotImplementedError
