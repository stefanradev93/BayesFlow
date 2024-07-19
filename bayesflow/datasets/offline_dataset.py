import keras
import math

from bayesflow.configurators import Configurator
from bayesflow.types import Tensor


class OfflineDataset(keras.utils.PyDataset):
    """
    A dataset that is pre-simulated and stored in memory.
    """

    def __init__(self, data: dict, batch_size: int, configurator: Configurator, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.configurator = configurator
        self.data = data
        self.indices = keras.ops.arange(len(data[next(iter(data.keys()))]), dtype="int64")

        self.shuffle()

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        """Get a batch of pre-simulated data"""
        if not 0 <= item < self.num_batches:
            raise IndexError(f"Index {item} is out of bounds for dataset with {self.num_batches} batches.")

        item = slice(item * self.batch_size, (item + 1) * self.batch_size)
        item = self.indices[item]

        batch = {key: keras.ops.take(value, item, axis=0) for key, value in self.data.items()}

        if self.configurator is not None:
            batch = self.configurator.configure(batch)

        return batch

    @property
    def num_batches(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def on_epoch_end(self) -> None:
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle the dataset in-place."""
        self.indices = keras.random.shuffle(self.indices)
