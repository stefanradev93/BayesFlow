from collections.abc import Mapping, MutableSequence
from copy import copy
import keras

from bayesflow.types import Tensor
from bayesflow.utils import expand_right_as, optimal_transport

from .configurator import Configurator


DataT = Mapping[str, Tensor]
VarT = MutableSequence[Tensor | None]


class FlowMatchingConfigurator(Configurator[DataT, VarT]):
    def __init__(self, inner: Configurator, index: int = 0, **kwargs):
        self.inner = inner
        self.index = index
        self.seed_generator = keras.random.SeedGenerator()
        self.kwargs = kwargs

    def configure(self, data: DataT) -> VarT:
        variables = self.inner.configure(data)

        x1 = variables[self.index]
        x0 = keras.random.normal(keras.ops.shape(x1), dtype=keras.ops.dtype(x1), seed=self.seed_generator)

        x0, x1 = optimal_transport(x0, x1, **self.kwargs)

        t = keras.random.uniform((keras.ops.shape(x0)[0],), seed=self.seed_generator)
        t = expand_right_as(t, x0)

        x = t * x1 + (1 - t) * x0
        v = x1 - x0

        variables[self.index] = (x, v)

        return variables

    def deconfigure(self, variables: VarT) -> DataT:
        if not isinstance(variables[self.index], tuple):
            return self.inner.deconfigure(variables)

        variables = copy(variables)
        x0, x1 = variables[self.index]
        variables[self.index] = x1

        return self.inner.deconfigure(variables)
