
import keras

from bayesflow.experimental.backend_agnostic.simulation.generative_model import GenerativeModel


def nested_getitem(data, item):
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = nested_getitem(value, item)
        else:
            result[key] = value[item]
    return result


class OfflineDataset(keras.utils.PyDataset):
    """
    A dataset that is pre-simulated and stored in memory.
    """
    def __init__(self, generative_model: GenerativeModel, batch_size: int, batches_per_epoch: int, **kwargs):
        super().__init__(**kwargs)
        self.generative_model = generative_model
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        self.data = self.generative_model.sample((batch_size * batches_per_epoch,))
        self.indices = keras.ops.arange(batch_size * batches_per_epoch)

        self.shuffle()

    def __getitem__(self, item: int) -> tuple:
        """ Get a batch of pre-simulated data """
        item = slice(item * self.batch_size, (item + 1) * self.batch_size)
        item = self.indices[item]
        data = nested_getitem(self.data, item)
        return (data,)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def on_epoch_end(self) -> None:
        self.shuffle()

    def shuffle(self) -> None:
        """ Shuffle the dataset in-place. """
        self.indices = keras.random.shuffle(self.indices)
