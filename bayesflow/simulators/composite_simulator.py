from collections.abc import Sequence

from bayesflow.types import Shape, Tensor

from .simulator import Simulator


class CompositeSimulator(Simulator):
    def __init__(self, simulators: Sequence[Simulator]):
        self.simulators = simulators

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, Tensor]:
        data = {}
        for simulator in self.simulators:
            data |= simulator.sample(batch_shape, **(kwargs | data))

        return data
