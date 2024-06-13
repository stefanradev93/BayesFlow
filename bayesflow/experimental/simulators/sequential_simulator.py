
from typing import Sequence

from bayesflow.experimental.types import Sampler, Shape, Tensor
from bayesflow.experimental.utils import batched_call

from .simulator import Simulator


class SequentialSimulator(Simulator):
    r"""
    Implements a sequentially factorized simulator:

    .. math::
        p(x) = \prod_{i = 1}^{n - 1} p(x_{i} | x_{i + 1}, ..., x_{n}) p(x_{n}


    Examples:
        >>> import numpy as np
        >>> def sample_contexts():
        >>>     return dict(contexts=np.random.normal())
        >>> def sample_parameters(shape: Shape, **kwargs):
        >>>     return dict(parameters=np.random.normal())
        >>> def sample_observables(contexts: Tensor, parameters: Tensor, **kwargs):
        >>>     observables = contexts + parameters + np.random.normal()
        >>>     return dict(observables=observables)
        >>> simulator = SequentialSimulator([sample_contexts, sample_parameters, sample_observables])
        >>> simulator.sample((2,))
        {'contexts': tensor(..., shape=(2, 1)), 'parameters': tensor(..., shape=(2, 1)), 'observables': tensor(..., shape=(2, 1))}

    """
    def __init__(self, samplers: Sequence[Sampler]):
        super().__init__()
        self.samplers = list(samplers)

    def sample(self, shape: Shape) -> dict[str, Tensor]:
        data = {}
        for sampler in self.samplers:
            data |= batched_call(sampler, shape[0], **data)

        return data
