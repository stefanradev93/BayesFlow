from collections.abc import Sequence
import keras

from bayesflow.types import Shape, Tensor

from .simulator import Simulator
from .composite_simulator import CompositeSimulator
from .lambda_simulator import LambdaSimulator


class SequentialSimulator(Simulator):
    def __init__(self, sample_fns: Sequence[callable], **kwargs):
        self.inner = CompositeSimulator([LambdaSimulator(fn, **kwargs) for fn in sample_fns])

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, Tensor]:
        data = self.inner.sample(batch_shape, **kwargs)

        for key, value in data.items():
            if keras.ops.ndim(value) == 1:
                value = keras.ops.expand_dims(value, axis=-1)

            data[key] = value

        return data
