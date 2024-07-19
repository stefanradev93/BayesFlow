from collections.abc import Sequence
import keras

from bayesflow.types import Shape, Tensor

from .simulator import Simulator


class CompositeSimulator(Simulator):
    """Combines multiple simulators into one, sequentially."""

    def __init__(self, simulators: Sequence[Simulator], expand_outputs: bool = False):
        self.simulators = simulators
        self.expand_outputs = expand_outputs

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, Tensor]:
        data = {}
        for simulator in self.simulators:
            data |= simulator.sample(batch_shape, **(kwargs | data))

        if self.expand_outputs:
            data = {
                key: keras.ops.expand_dims(value, axis=-1) if keras.ops.ndim(value) == 1 else value
                for key, value in data.items()
            }

        return data
