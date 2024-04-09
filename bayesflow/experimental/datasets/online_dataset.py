
import keras

from bayesflow.experimental.simulation.distributions.generative_model import GenerativeModel
from bayesflow.experimental.types import Data


class OnlineDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly.
    """
    def __init__(self, generative_model: GenerativeModel, batch_size: int, batches_per_epoch: int, **kwargs):
        super().__init__(**kwargs)
        self.generative_model = generative_model
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def __getitem__(self, item: int) -> (Data,):
        """ Sample a batch of data from the generative model """
        data = self.generative_model.sample((self.batch_size,))
        return (data,)

    def __len__(self) -> int:
        return self.batches_per_epoch
