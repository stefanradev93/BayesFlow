from collections.abc import Sequence

from bayesflow.types import Shape, Tensor

from .simulator import Simulator
from .composite_simulator import CompositeSimulator
from .lambda_simulator import LambdaSimulator


class CompositeLambdaSimulator(Simulator):
    """Combines multiple lambda simulators into one, sequentially."""

    def __init__(self, sample_fns: Sequence[callable], expand_outputs: bool = True, **kwargs):
        self.inner = CompositeSimulator(
            [LambdaSimulator(fn, **kwargs) for fn in sample_fns], expand_outputs=expand_outputs
        )

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, Tensor]:
        return self.inner.sample(batch_shape, **kwargs)
