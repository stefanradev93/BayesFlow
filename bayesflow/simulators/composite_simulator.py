from collections.abc import Mapping
import numpy as np

from bayesflow.types import Shape

from .simulator import Simulator


class CompositeSimulator(Simulator):
    """Combines multiple simulators into one, sequentially."""

    def __init__(
        self,
        simulators: Mapping[str, Simulator],
        expand_outputs: bool = False,
        global_fns: Mapping[str, callable] = None,
    ):
        self.simulators = simulators
        self.expand_outputs = expand_outputs
        self.global_fns = global_fns if global_fns else {}

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        data = {}
        global_vars = {}
        for name, simulator in self.simulators.items():
            global_var = self.global_fns[name]() if self.global_fns.get(name) is not None else {}
            data |= simulator.sample(batch_shape, **(kwargs | global_var | data))
            global_vars |= global_var

        global_vars = {
            key: np.repeat(np.expand_dims(value, axis=0), batch_shape[0], axis=0) for key, value in global_vars.items()
        }

        if self.expand_outputs:
            data = {
                key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value for key, value in data.items()
            }

            global_vars = {
                key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value
                for key, value in global_vars.items()
            }

        data |= global_vars

        return data
