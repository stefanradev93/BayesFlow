from collections.abc import Sequence
import keras

from .composite_simulator import Compose
from .functional_simulator import FunctionalSimulator
from .simulator import Simulator
from ..types import Shape, Tensor


class SequentialSimulator(Simulator):
    def __init__(self, sample_fns: Sequence[callable], *, convert_dtypes: str = None, **kwargs):
        self.inner = Compose([FunctionalSimulator(fn, **kwargs) for fn in sample_fns])
        self.convert_dtypes = convert_dtypes

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, Tensor]:
        data = self.inner.sample(batch_shape, **kwargs)

        for key, value in data.items():
            if keras.ops.ndim(value) == 1:
                value = keras.ops.expand_dims(value, axis=-1)

            data[key] = value

        return data
