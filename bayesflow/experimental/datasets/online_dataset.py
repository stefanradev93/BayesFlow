
import keras

from bayesflow.experimental.configurator import Configurator
from bayesflow.experimental.types import Data
from bayesflow.experimental.simulation.distributions.generative_model import GenerativeModel


class OnlineDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly.
    """
    def __init__(self, generative_model: GenerativeModel, batch_size: int, batches_per_epoch: int, configurator: Configurator = Configurator(), **kwargs):
        super().__init__(**kwargs)
        self.generative_model = generative_model
        self.configurator = configurator
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def __getitem__(self, item: int) -> (Data,):
        """ Sample a batch of data from the generative model """
        data = self.generative_model.sample((self.batch_size,))
        data = self.configurator(data)
        return (data,)

    def __len__(self) -> int:
        return self.batches_per_epoch
