from collections.abc import Mapping
import numpy as np

from bayesflow.types import Shape

from .simulator import Simulator
from .composite_simulator import CompositeSimulator
from .lambda_simulator import LambdaSimulator


class CompositeLambdaSimulator(Simulator):
    """Combines multiple lambda simulators into one, sequentially."""

    def __init__(
        self,
        sample_fns: Mapping[str, callable],
        expand_outputs: bool = True,
        global_fns: Mapping[str, callable] = None,
        **kwargs,
    ):
        # TODO Validate case where global_fns is not None and not part of dict
        self.inner = CompositeSimulator(
            simulators={name: LambdaSimulator(fn, **kwargs) for name, fn in sample_fns.items()},
            expand_outputs=expand_outputs,
            global_fns=global_fns,
        )

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        return self.inner.sample(batch_shape, **kwargs)
