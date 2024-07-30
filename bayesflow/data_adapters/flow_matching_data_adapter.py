import keras

from bayesflow.types import Tensor
from bayesflow.utils import optimal_transport

from .data_adapter import DataAdapter


TRaw = any
TProcessedInner = dict[str, Tensor]
TProcessed = dict[str, Tensor | tuple[Tensor, ...]]


class FlowMatchingDataAdapter(DataAdapter[TRaw, TProcessed]):
    """Wraps a data adapter, applying all further processing required for Optimal Transport Flow Matching.
    Useful to move these operations into a worker process, so as not to slow down training.
    """

    def __init__(self, inner: DataAdapter[TRaw, TProcessedInner], key: str = "inference_variables", **kwargs):
        self.inner = inner
        self.key = key
        self.kwargs = kwargs

        self.seed_generator = keras.random.SeedGenerator()

    def configure(self, raw_data: TRaw) -> TProcessed:
        processed_data = self.inner.configure(raw_data)

        x1: Tensor = processed_data[self.key]
        x0: Tensor = keras.random.normal(keras.ops.shape(x1), dtype=keras.ops.dtype(x1), seed=self.seed_generator)
        t: Tensor = keras.random.uniform(keras.ops.shape(x1)[0], dtype=keras.ops.dtype(x1), seed=self.seed_generator)

        x0, x1 = optimal_transport(x0, x1, **self.kwargs)

        x: Tensor = t * x1 + (1 - t) * x0

        target_velocity: Tensor = x1 - x0

        return processed_data | {self.key: (x0, x1, t, x, target_velocity)}

    def deconfigure(self, variables: TProcessed) -> TRaw:
        return self.inner.deconfigure(variables)
