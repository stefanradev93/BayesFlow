from collections.abc import Sequence
import keras
import numpy as np

from bayesflow.types import Shape, Tensor
from bayesflow.utils import expand_tile, tree_stack

from .simulator import Simulator


class ModelComparisonSimulator(Simulator):
    """Wraps a sequence of simulators for use with a model comparison approximator."""

    def __init__(
        self,
        simulators: Sequence[Simulator],
        logits: Sequence[float] = None,
        use_mixed_batches: bool = False,
    ):
        # TODO: use simulator_probabilities instead of logits
        self.simulators = simulators
        self.logits = logits or [0.0] * len(simulators)
        if len(logits) != len(simulators):
            raise ValueError(f"Length of logits ({len(logits)}) must match number of simulators ({len(simulators)}).")

        self.logits = keras.ops.convert_to_tensor(self.logits)
        self.seed_generator = keras.random.SeedGenerator()
        self.use_mixed_batches = use_mixed_batches

    def sample(self, batch_shape: Shape, *, numpy: bool = False, **kwargs) -> dict[str, Tensor]:
        if not self.use_mixed_batches:
            # draw one model index for the whole batch (faster)
            logits = keras.ops.expand_dims(self.logits, 0)
            model_indices = int(keras.random.categorical(logits, 1, seed=self.seed_generator))

            simulator = self.simulators[model_indices]
            data = simulator.sample(batch_shape)

            model_indices = keras.ops.full(batch_shape, model_indices, dtype="int32")
        else:
            # draw a model index for each sample in the batch (slower)
            batch_size = np.prod(batch_shape)
            logits = expand_tile(self.logits, axis=0, n=batch_size)
            model_indices = keras.random.categorical(logits, 1, seed=self.seed_generator)

            data = [self.simulators[int(i)].sample(()) for i in model_indices]
            data = tree_stack(data, axis=0)

            # restore batch shape
            data = {
                key: keras.ops.reshape(value, batch_shape + keras.ops.shape(value)[1:]) for key, value in data.items()
            }

        model_indices = keras.ops.one_hot(model_indices, len(self.simulators))

        if numpy:
            data = {key: keras.ops.convert_to_numpy(value) for key, value in data.items()}
            model_indices = keras.ops.convert_to_numpy(model_indices)

        return {"model_indices": model_indices, **data}
